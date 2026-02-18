Core Frameworks
These open-source tools enable agentic swarms tailored for web scraping.

Framework	Key Features	Web Scraping Fit	GitHub Stars (approx.)	Deployment Notes
OpenAI Swarm	Multi-agent handoffs, LLM orchestration, routines for dynamic tasks	Agents for URL fetching, form filling, data extraction to JSON/markdown for training	High (experimental but widely forked) 
​
​	Dockerize Python scripts; scale via Kubernetes 
​
kyegomez/swarms	Enterprise-grade multi-agent platform, tool integrations, production scaling	Parallel scraping agents, anti-bot handling, structured output for datasets	Production-focused 
​	Built-in Docker/K8s support for swarms 
​
ScrapeGraphAI/Scrapecraft	AI-powered scraping pipelines, LangGraph orchestration, schema generation	Bulk URL scraping, form data extraction, LLM-ready outputs without boilerplate	Agentic for forms/pipelines 
​	Dockerized, easy local/prod deploy 
​
LangGraph (LangChain)	Stateful multi-agent graphs, error recovery, custom tools	Web traversal, form submission, data validation for training sets	Mature ecosystem 
Helm charts or Docker Compose for swarms 
​
Deployment Tools
Deploy swarms scalably using these Kubernetes-native open-source platforms, aligning with your GCP/Docker expertise.
​

Kagent: CNCF-incubating framework for running AI agent swarms in K8s; automates ops with MCP tools (e.g., browser access). Quick Helm install for production scraping clusters.

ARK (Agentic Runtime for Kubernetes): CRDs for agents/teams/tools/memory; deploy scraping swarms as K8s resources with persistence.
​

Agent Sandbox (Kubernetes SIG): Secure isolation for agent code (e.g., untrusted scraping scripts); scales to thousands of parallel form-filling agents.