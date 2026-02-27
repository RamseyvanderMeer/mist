"""
Conversational RAG orchestrator for multi-turn diagnostic conversations.
"""
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging

from src.paths import Paths
from src.retrieval.enhanced_retriever import EnhancedRetriever, EnhancedRetrieverError
from src.retrieval.ambiguity_detector import AmbiguityDetector, AmbiguityDetectorError
from src.retrieval.clarification_generator import ClarificationGenerator, ClarificationGeneratorError
from src.retrieval.query_expander import QueryExpander, QueryExpansionError
from src.retrieval.session_manager import SessionManager

logger = logging.getLogger(__name__)


class ConversationalRAGError(Exception):
    """Base exception for ConversationalRAG errors."""
    pass


class ConversationalRAG:
    """
    Main orchestrator for conversational retrieval-augmented generation.
    
    Integrates EnhancedRetriever, AmbiguityDetector, ClarificationGenerator,
    QueryExpander, and SessionManager to provide a unified interface for
    multi-turn diagnostic conversations.
    """
    
    def __init__(self, retrieval_config_path: Optional[Union[str, Path]] = None, llm_config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConversationalRAG with all required components.
        
        Args:
            retrieval_config_path: Optional path to retrieval_config.yaml.
                                 If None, uses default from Paths.
            llm_config_path: Optional path to llm_config.yaml.
                            If None, uses default from Paths.
        
        Raises:
            ConversationalRAGError: If initialization fails
        """
        try:
            # Get paths
            paths = Paths()
            
            # Determine config paths
            if retrieval_config_path is None:
                retrieval_config_path = paths.retrieval_config
            else:
                retrieval_config_path = Path(retrieval_config_path)
            
            if llm_config_path is None:
                llm_config_path = paths.llm_config
            else:
                llm_config_path = Path(llm_config_path)
            
            # Initialize components
            logger.info("Initializing ConversationalRAG components...")
            
            # EnhancedRetriever (handles multi-stage retrieval)
            self.retriever = EnhancedRetriever(config_path=retrieval_config_path)
            
            # AmbiguityDetector (checks if clarification needed)
            self.ambiguity_detector = AmbiguityDetector(config_path=retrieval_config_path)
            
            # ClarificationGenerator (generates clarification questions)
            self.clarification_generator = ClarificationGenerator(config_path=str(llm_config_path))
            
            # QueryExpander (expands queries with user responses)
            self.query_expander = QueryExpander(config_path=str(llm_config_path))
            
            # SessionManager (manages multi-turn conversations)
            self.session_manager = SessionManager()
            
            # Load clarification config from retrieval config
            import yaml
            with open(retrieval_config_path, 'r', encoding='utf-8') as f:
                retrieval_config = yaml.safe_load(f)
            
            clarification_config = retrieval_config.get("clarification", {})
            self.clarification_enabled = clarification_config.get("enabled", True)
            self.max_questions = clarification_config.get("max_questions", 3)
            self.max_clarifications_per_session = clarification_config.get("max_clarifications_per_session", 3)
            
            logger.info(
                f"Initialized ConversationalRAG: "
                f"clarification_enabled={self.clarification_enabled}, "
                f"max_questions={self.max_questions}, "
                f"max_clarifications_per_session={self.max_clarifications_per_session}"
            )
            
        except EnhancedRetrieverError as e:
            raise ConversationalRAGError(f"Failed to initialize EnhancedRetriever: {e}") from e
        except AmbiguityDetectorError as e:
            raise ConversationalRAGError(f"Failed to initialize AmbiguityDetector: {e}") from e
        except ClarificationGeneratorError as e:
            raise ConversationalRAGError(f"Failed to initialize ClarificationGenerator: {e}") from e
        except QueryExpansionError as e:
            raise ConversationalRAGError(f"Failed to initialize QueryExpander: {e}") from e
        except Exception as e:
            raise ConversationalRAGError(f"Failed to initialize ConversationalRAG: {e}") from e
    
    def query(
        self,
        fault_codes: List[str],
        obd_data: Dict[str, Any],
        description: Optional[str] = None,
        vehicle_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process initial query with fault codes, OBD data, and optional problem description.
        
        Orchestrates the full retrieval pipeline:
        1. Session management (create or retrieve)
        2. Multi-stage retrieval
        3. Ambiguity detection
        4. Clarification generation (if needed)
        5. Session state updates
        
        Args:
            fault_codes: List of fault code strings (e.g., ["P0301", "P0302"])
            obd_data: OBD sensor data dictionary
            description: Optional problem/symptom description for semantic search
            vehicle_context: Optional vehicle information (merged into obd_data)
            session_id: Optional session ID for continuing existing conversation
        
        Returns:
            Dict with structure:
            - recommendations: List of recommendation dicts
            - needs_clarification: bool
            - clarification_questions: Optional[List[str]]
            - session_id: str
            - query_text: str
        """
        try:
            # Expand symptom description to bridge symptom->fix semantic gap
            # (e.g. "engine too hot" -> "overheating, coolant, radiator, thermostat")
            expanded_description = description
            if description and description.strip():
                try:
                    expanded_description = self.query_expander.expand_symptom_for_search(
                        description.strip()
                    )
                    if expanded_description != description:
                        logger.info(
                            f"Symptom expansion applied for session (original: {len(description)} chars, "
                            f"expanded: {len(expanded_description)} chars)"
                        )
                except Exception as e:
                    logger.warning(f"Symptom expansion failed, using original: {e}")
                    expanded_description = description

            # Generate query text from fault codes and (expanded) description
            query_text = self._generate_query_text(fault_codes, expanded_description)
            
            # Merge vehicle_context into obd_data if provided
            if vehicle_context:
                obd_data = {**obd_data, **vehicle_context}
            
            # Session management
            if session_id:
                # Retrieve existing session
                session_data = self.session_manager.get_session(session_id)
                if session_data is None:
                    logger.warning(f"Session {session_id} not found, creating new session")
                    session_id = None
            
            if session_id is None:
                # Create new session
                session_id = self.session_manager.create_session(
                    fault_codes=fault_codes,
                    obd_data=obd_data,
                    vehicle_context=vehicle_context,
                    description=description
                )
                logger.info(f"Created new session: {session_id}")
            else:
                logger.info(f"Continuing session: {session_id}")
            
            # Multi-stage retrieval (use expanded description for better symptom->fix matching)
            try:
                ranked_results = self.retriever.retrieve(
                    fault_codes=fault_codes,
                    obd_data=obd_data,
                    query_text=query_text,
                    description=expanded_description
                )
                logger.info(f"Retrieved {len(ranked_results)} candidates for session {session_id}")
            except EnhancedRetrieverError as e:
                logger.error(f"Retrieval failed for session {session_id}: {e}")
                ranked_results = []
            
            # Format recommendations
            recommendations = self._format_recommendations(ranked_results)
            
            # Ambiguity detection
            needs_clarification = False
            clarification_questions: Optional[List[str]] = None
            
            if self.clarification_enabled and ranked_results:
                try:
                    is_ambiguous, reason = self.ambiguity_detector.detect(
                        ranked_results=ranked_results,
                        obd_data=obd_data
                    )
                    
                    if is_ambiguous:
                        logger.debug(f"Ambiguity detected for session {session_id}: {reason}")
                        
                        # Check if clarification limit exceeded
                        if not self._check_clarification_limit(session_id):
                            needs_clarification = True
                            
                            # Generate clarification questions
                            try:
                                questions = self.clarification_generator.generate_questions(
                                    fault_codes=fault_codes,
                                    obd_data=obd_data,
                                    ranked_candidates=ranked_results[:3]  # Top 3 for context
                                )
                                
                                # Limit to max_questions
                                clarification_questions = questions[:self.max_questions]
                                
                                logger.info(
                                    f"Generated {len(clarification_questions)} clarification questions "
                                    f"for session {session_id}"
                                )
                            except ClarificationGeneratorError as e:
                                logger.error(f"Failed to generate clarification questions: {e}")
                                clarification_questions = []
                        else:
                            logger.info(
                                f"Clarification limit exceeded for session {session_id}, "
                                f"skipping clarification generation"
                            )
                    else:
                        logger.debug(f"No ambiguity detected for session {session_id}")
                        
                except AmbiguityDetectorError as e:
                    logger.error(f"Ambiguity detection failed: {e}")
                    # Continue without clarification
            
            # Update session with recommendations and questions
            try:
                procedure_ids = [
                    rec.get("procedure_id", "") 
                    for rec in recommendations 
                    if rec.get("procedure_id")
                ]
                self.session_manager.update_session(
                    session_id=session_id,
                    recommended_guides=procedure_ids
                )
                
                if clarification_questions:
                    # Store clarification questions (will be updated with responses later)
                    self.session_manager.update_session(
                        session_id=session_id,
                        clarification_questions=clarification_questions
                    )
                    
            except Exception as e:
                logger.error(f"Failed to update session {session_id}: {e}")
                # Continue - session update failure shouldn't block response
            
            # Build response
            response = {
                "recommendations": recommendations,
                "needs_clarification": needs_clarification,
                "clarification_questions": clarification_questions,
                "session_id": session_id,
                "query_text": query_text
            }
            
            logger.info(
                f"Query completed for session {session_id}: "
                f"{len(recommendations)} recommendations, "
                f"clarification={'needed' if needs_clarification else 'not needed'}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error in query() for session {session_id}: {e}", exc_info=True)
            # Return partial response with error indication
            return {
                "recommendations": [],
                "needs_clarification": False,
                "clarification_questions": None,
                "session_id": session_id or "error",
                "query_text": self._generate_query_text(fault_codes) if fault_codes else ""
            }
    
    def clarify(self, session_id: str, responses: List[str]) -> Dict[str, Any]:
        """
        Process clarification responses and refine recommendations.
        
        Retrieves session, expands query with user responses, re-runs retrieval,
        and updates session state.
        
        Args:
            session_id: Session ID from initial query
            responses: List of user responses to clarification questions
        
        Returns:
            Dict with structure:
            - recommendations: List of refined recommendation dicts
            - needs_clarification: bool (typically False after clarification)
            - clarification_questions: Optional[List[str]] (None after clarification)
            - session_id: str
            - query_text: str (expanded query)
        
        Raises:
            ConversationalRAGError: If session not found or critical error occurs
        """
        try:
            # Retrieve session
            session_data = self.session_manager.get_session(session_id)
            if session_data is None:
                raise ConversationalRAGError(f"Session {session_id} not found")
            
            logger.info(f"Processing clarification for session {session_id} with {len(responses)} responses")
            
            # Get original session data
            fault_codes = session_data.get("fault_codes", [])
            obd_data = session_data.get("obd_data", {})
            description = session_data.get("description")
            existing_questions = session_data.get("clarification_questions", [])
            
            # Validate responses match questions
            if len(responses) != len(existing_questions):
                logger.warning(
                    f"Response count ({len(responses)}) doesn't match question count "
                    f"({len(existing_questions)}) for session {session_id}"
                )
            
            # Generate original query text (include description for expansion context)
            original_query_text = self._generate_query_text(fault_codes, description)
            
            # Expand query with user responses
            expanded_query_text = original_query_text
            try:
                expanded_query_text = self.query_expander.expand_query(
                    original_query=original_query_text,
                    user_responses=responses
                )
                logger.info(
                    f"Expanded query for session {session_id}: "
                    f"'{original_query_text}' -> '{expanded_query_text[:100]}...'"
                )
            except QueryExpansionError as e:
                logger.warning(f"Query expansion failed for session {session_id}: {e}. Using original query.")
                expanded_query_text = original_query_text
            
            # Re-retrieve with expanded query (use expanded as search query)
            try:
                ranked_results = self.retriever.retrieve(
                    fault_codes=fault_codes,
                    obd_data=obd_data,
                    query_text=expanded_query_text,
                    search_query_text=expanded_query_text
                )
                logger.info(f"Re-retrieved {len(ranked_results)} candidates for session {session_id}")
            except EnhancedRetrieverError as e:
                logger.error(f"Re-retrieval failed for session {session_id}: {e}")
                ranked_results = []
            
            # Format recommendations
            recommendations = self._format_recommendations(ranked_results)
            
            # Update session with clarification round
            try:
                self.session_manager.add_clarification_round(
                    session_id=session_id,
                    questions=existing_questions[:len(responses)],  # Match response count
                    responses=responses
                )
                
                # Update recommendations
                procedure_ids = [
                    rec.get("procedure_id", "")
                    for rec in recommendations
                    if rec.get("procedure_id")
                ]
                self.session_manager.update_recommendations(
                    session_id=session_id,
                    recommended_guides=procedure_ids
                )
                
                logger.info(f"Updated session {session_id} with clarification round")
            except Exception as e:
                logger.error(f"Failed to update session {session_id}: {e}")
                # Continue - session update failure shouldn't block response
            
            # Build response (typically no further clarification needed after user responses)
            response = {
                "recommendations": recommendations,
                "needs_clarification": False,  # After clarification, typically no more needed
                "clarification_questions": None,
                "session_id": session_id,
                "query_text": expanded_query_text
            }
            
            logger.info(
                f"Clarification completed for session {session_id}: "
                f"{len(recommendations)} refined recommendations"
            )
            
            return response
            
        except ConversationalRAGError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in clarify() for session {session_id}: {e}", exc_info=True)
            raise ConversationalRAGError(f"Failed to process clarification: {e}") from e
    
    def _generate_query_text(
        self,
        fault_codes: List[str],
        description: Optional[str] = None
    ) -> str:
        """
        Generate query text from fault codes and optional description.
        
        Args:
            fault_codes: List of fault code strings
            description: Optional problem/symptom description
        
        Returns:
            Query text string for search and re-ranking
        """
        if not fault_codes:
            return description.strip() if description else ""
        fault_text = ", ".join(fault_codes)
        if description and description.strip():
            return f"Fault codes: {fault_text}. Problem: {description.strip()}"
        return fault_text
    
    def _format_recommendations(self, ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format EnhancedRetriever results to match response schema.
        
        Args:
            ranked_results: List of result dicts from EnhancedRetriever.retrieve()
        
        Returns:
            List of formatted recommendation dicts with structure:
            - id: str (procedure_id or document id)
            - title: str
            - procedure_name: str
            - procedure_id: Optional[str]
            - score: float (combined_score)
            - text: Optional[str]
        """
        formatted = []
        
        for result in ranked_results:
            # Extract fields from EnhancedRetriever result
            procedure_id = result.get("procedure_id") or result.get("id", "")
            title = result.get("title") or result.get("procedure_name", "Unknown")
            procedure_name = result.get("procedure_name") or title
            score = result.get("combined_score", result.get("score", 0.0))
            text = result.get("text")
            
            formatted.append({
                "id": procedure_id or result.get("id", ""),
                "title": title,
                "procedure_name": procedure_name,
                "procedure_id": procedure_id if procedure_id else None,
                "score": float(score),
                "text": text
            })
        
        return formatted
    
    def _check_clarification_limit(self, session_id: str) -> bool:
        """
        Check if max clarifications per session has been exceeded.
        
        Since SessionManager stores questions and responses in arrays that grow
        with each clarification round, we estimate rounds by checking if responses
        exist. Each completed clarification round adds responses via add_clarification_round.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if limit exceeded, False otherwise
        """
        try:
            session_data = self.session_manager.get_session(session_id)
            if session_data is None:
                return False  # New session, no limit exceeded
            
            # Get existing clarification data
            existing_responses = session_data.get("user_responses", [])
            existing_questions = session_data.get("clarification_questions", [])
            
            # Count completed clarification rounds
            # Each call to add_clarification_round completes one round
            # Since responses are only added when a round is completed,
            # we can estimate rounds by checking response count
            # However, each round can have multiple responses (1-3 questions typically)
            # So we need to estimate conservatively
            
            # Simple heuristic: if we have responses, we've completed at least one round
            # For a more accurate count, we'd need to track rounds explicitly
            # For now, use a conservative estimate: count rounds as 1 if any responses exist
            # This prevents unlimited clarifications while being permissive
            
            # More accurate: estimate based on typical round size
            # If max_questions is 3, and we have 6 responses, that's ~2 rounds
            # But this is still an estimate
            
            # Simplest working approach: if responses exist, we've completed rounds
            # Count conservatively to avoid blocking legitimate clarifications
            rounds_completed = 1 if existing_responses else 0
            
            # If we have many responses, estimate multiple rounds
            # Assume average of max_questions per round
            if existing_responses and self.max_questions > 0:
                estimated_rounds = max(1, len(existing_responses) // self.max_questions)
                rounds_completed = estimated_rounds
            
            exceeded = rounds_completed >= self.max_clarifications_per_session
            
            if exceeded:
                logger.debug(
                    f"Clarification limit exceeded for session {session_id}: "
                    f"estimated {rounds_completed} rounds >= {self.max_clarifications_per_session}"
                )
            
            return exceeded
            
        except Exception as e:
            logger.error(f"Error checking clarification limit for session {session_id}: {e}")
            return False  # On error, allow clarification (fail open)
