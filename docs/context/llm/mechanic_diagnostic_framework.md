# Mechanic Diagnostic Questions Framework

This document provides the diagnostic question framework used by the MIST API's query expansion system to generate clarifying questions for users.

## Overview

When a user describes a vehicle issue (with or without fault codes), the system uses this framework to ask targeted questions that help narrow down the diagnosis and match to the correct repair guide from our database of 337,000+ BMW procedures.

## Question Categories

### 1. Vehicle Context
Questions to establish the baseline vehicle information:

- "What is the year, make, and model of your vehicle?"
- "What is the current mileage?"
- "What engine does it have?"
- "Are there any modifications (tuning, exhaust, suspension, etc.)?"

**Why:** Different model years, engines, and modifications have different common failure points and diagnostic procedures.

### 2. Problem Timeline & History
Questions to understand the problem's progression:

- "When did you first notice this issue?"
- "Did it start suddenly or gradually get worse?"
- "Has the vehicle been in any accidents or had recent repairs?"
- "Did this start after a specific event (pothole, flood, battery change, etc.)?"

**Why:** Sudden vs gradual onset indicates different failure modes. Trigger events help identify root causes.

### 3. Operating Conditions (Most Diagnostic)
Questions about when and how the problem occurs:

- "Under what conditions does it occur - accelerating, cruising, braking, turning, or idling?"
- "Does it happen when the engine is cold, warming up, or fully hot?"
- "Is it at low speed, highway speed, or specific RPM?"
- "Does weather matter - rain, cold, hot, or damp conditions?"
- "Does it happen only in stop-and-go traffic, highway, or both?"
- "Does it tend to start after the vehicle has been driven a certain distance?"

**Why:** These are the MOST DIAGNOSTIC questions. Different systems fail under different conditions (e.g., ignition issues when cold, overheating when hot, CV joints when turning).

### 4. Symptom Details
Questions to characterize what the user experiences:

- "What exactly do you notice - noise, vibration, warning lights, loss of power, smell, or leak?"
- "Can you describe any noises - squeal, grind, knock, click, rattle, hum, whine?"
- "Is the noise high or low pitched, constant or intermittent?"
- "Where does it seem to come from - engine bay, wheels, steering, brakes?"

**Why:** Specific symptom types point to specific systems. Noise characteristics help isolate mechanical vs electrical issues.

### 5. Warning Lights & Gauges
Questions about dashboard indicators:

- "What warning lights are on (check engine, ABS, airbag, etc.)?"
- "Have you noticed changes in temperature, oil pressure, or other gauges?"
- "What exactly does the dash display say?"

**Why:** Warning lights and gauge readings provide direct diagnostic codes and system status information.

## Implementation Notes

### For Symptom-Only Queries
When a user provides only a symptom description (no fault codes):
1. Use the symptom_clarification prompt template
2. Generate 2-4 targeted questions from the framework above
3. Prioritize Operating Conditions questions - these are most diagnostic
4. Do NOT ask about physical test drives (we cannot do that remotely)

### For Fault Code Queries
When fault codes are available:
1. Use the clarification prompt template
2. Analyze the fault codes and top repair guide candidates
3. Ask 1-3 questions that would help distinguish between the top candidates
4. Focus on missing information that would confirm one repair over another

### Question Selection Strategy
The LLM should select questions based on:
1. **Information Gap**: What critical details are missing?
2. **Diagnostic Value**: Which questions would most narrow down the diagnosis?
3. **User Accessibility**: Can the user reasonably answer this without special tools?
4. **Repair Guide Matching**: Would the answer help match to a specific procedure?

## Example Question Flows

### Example 1: "My engine is running rough"
1. "When does it happen - at idle, when accelerating, or at highway speeds?"
2. "Is the check engine light on, and if so, is it flashing or steady?"
3. "Did this start suddenly or has it been getting worse over time?"

### Example 2: "There's a noise when I drive"
1. "What type of noise - squeal, grind, rattle, hum, or knock?"
2. "When do you hear it - when turning, braking, accelerating, or at constant speed?"
3. "Does it change with vehicle speed or engine speed?"

### Example 3: P0301 (Cylinder 1 Misfire) + "engine shakes at idle"
1. "Does the shaking get worse when the engine is cold or when it's fully warmed up?"
2. "Have you noticed any changes in fuel economy or exhaust smell?"

## Integration with MIST API

The framework is integrated into:
- `config/llm_config.yaml` - Prompt templates with the framework as system context
- `src/llm/prompt_templates.py` - Template loading and variable substitution
- `src/retrieval/query_expander.py` - Question generation methods

The API uses this framework in the `/query` endpoint when `needs_clarification=true` is returned, and in the `/clarify` endpoint to process user responses.

---

*This framework is based on professional automotive diagnostic best practices, OEM service procedures, and real-world shop experience.*
