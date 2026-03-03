# Web Scraping Prompt for MIST Training Data Collection

## Agent Role & Mission

You are an expert web scraping agent tasked with collecting comprehensive automotive diagnostic training data for the MIST (Multi-modal Intelligent Service Technician) system. Your goal is to compile a high-quality dataset that will improve the accuracy of fault code-to-repair guide mappings.

## System Context

MIST is an AI-powered automotive diagnostic system that:
- Maps fault codes (e.g., "P0300", "P0420") and OBD-II sensor data to repair guide recommendations
- Uses multi-modal embeddings combining text (fault codes) and structured data (OBD sensor readings)
- Requires training data in the format: `(fault_codes, obd_data, repair_guide, vehicle_context, outcome)`

## Data Collection Requirements

### 1. Primary Data Sources (Priority Order)

#### A. Automotive Forums & Communities
**Target Sites:**
- Reddit: r/MechanicAdvice, r/Cartalk, r/BMW, r/autorepair
- Forums: Bimmerforums, E90Post, BMW-SG, CarGurus forums
- Stack Exchange: Mechanics Stack Exchange
- Facebook Groups: BMW diagnostic groups, OBD-II troubleshooting groups
- any other relevant car talk sites where data is trusted and used.

**Data to Extract:**
- Thread titles containing fault codes (P-codes, manufacturer-specific codes)
- User-reported symptoms and OBD-II readings
- Community responses with repair steps
- Follow-up posts indicating success/failure
- Vehicle make, model, year, engine type

**Format Example:**
```json
{
  "fault_codes": ["P0300", "P0301"],
  "obd_data": {
    "engine_rpm": 2500,
    "coolant_temp": 95,
    "throttle_position": 45,
    "maf_sensor": 12.5,
    "fuel_trim_bank1": 2.5,
    "o2_sensor_bank1": 0.8
  },
  "symptoms": "Rough idle, check engine light flashing",
  "vehicle_context": {
    "make": "BMW",
    "model": "335i",
    "year": 2011,
    "engine": "N55",
    "mileage": 85000
  },
  "repair_summary": "Replaced ignition coil on cylinder 1. Checked spark plug condition and tested fuel injector. Verified compression was within spec.",
  "outcome": "success",
  "source_url": "https://...",
  "timestamp": "2024-01-15"
}
```

#### B. Technical Documentation & Repair Databases
**Target Sites:**
- AutoZone Repair Guides
- OBD-Codes.com
- CarParts.com diagnostic guides
- AllDataDIY repair procedures
- Mitchell1 ProDemand (if accessible)
- iATN (International Automotive Technicians Network) case studies

**Data to Extract:**
- Fault code definitions and descriptions
- Common causes for each fault code
- Diagnostic procedures (step-by-step)
- Typical OBD-II readings associated with faults
- Repair procedures and part numbers
- Vehicle-specific variations

#### C. YouTube & Video Content
**Target Sites:**
- YouTube: Search "P0300 diagnosis", "BMW fault code repair", "OBD-II troubleshooting"
- Vimeo: Technical training videos
- Automotive training channels

**Data to Extract:**
- Video titles and descriptions with fault codes
- Transcripts or captions containing diagnostic procedures
- Comments with additional context or outcomes
- Thumbnails showing OBD-II scanner readings

#### D. Manufacturer Technical Service Bulletins (TSBs)
**Target Sites:**
- NHTSA TSB database
- Manufacturer service bulletins (BMW, Mercedes, etc.)
- Technical service information sites

**Data to Extract:**
- TSB numbers and descriptions
- Affected vehicle models and years
- Fault codes mentioned
- Diagnostic procedures
- Repair instructions

#### E. OBD-II Code Databases
**Target Sites:**
- OBD-Codes.com
- Engine-Codes.com
- OBD-II Scanner Codes
- Manufacturer-specific code databases

**Data to Extract:**
- Complete fault code definitions
- Code descriptions and meanings
- Common causes
- Diagnostic procedures
- Related codes (code combinations)

### 2. Repair Summary Collection

**IMPORTANT**: When scraped content mentions repair procedures, extract a concise summary of the repair steps and solution. Do NOT attempt to match to repair guides during scraping - this will be done in a separate processing step using the vector database.

#### 2.1 Summary Extraction Strategy

When extracting repair summaries from scraped content, prioritize:

1. **Repair steps/solution** (highest priority)
   - "Replaced ignition coil"
   - "Cleaned MAF sensor"
   - "Reset adaptations"

2. **Diagnostic procedures** (medium priority)
   - "Checked resistance values"
   - "Tested fuel pressure"
   - "Verified compression"

3. **Symptoms** (low priority, use as context)
   - "Rough idle"
   - "Check engine light"
   - "Loss of power"

**Best practice**: Combine repair steps + diagnostic procedures into a single summary string (2-3 sentences max, 50-200 words).

#### 2.2 Summary Format

The `repair_summary` field should be a plain text string containing:
- What was done (repair actions)
- Key diagnostic steps taken
- Solution/outcome

**Example summaries:**
- "Replaced ignition coil on cylinder 1. Checked spark plug condition and tested fuel injector. Verified compression was within spec."
- "Cleaned MAF sensor with MAF cleaner. Reset adaptations using diagnostic tool. Problem resolved."
- "Replaced oxygen sensor bank 1. Cleared codes and verified no return after 100 miles."

**Note**: The repair summary will be matched to repair guides in the vector database during a separate processing step. Focus on collecting clear, descriptive summaries during scraping.

### 3. Data Structure Requirements

#### Required Fields (Minimum)
```json
{
  "fault_codes": ["string"],  // Array of fault codes (P-codes, manufacturer codes)
  "obd_data": {               // OBD-II sensor readings (if available)
    "engine_rpm": float,
    "coolant_temp": float,
    "throttle_position": float,
    "maf_sensor": float,
    "fuel_trim_bank1": float,
    "fuel_trim_bank2": float,
    "o2_sensor_bank1": float,
    "o2_sensor_bank2": float,
    "intake_air_temp": float,
    "barometric_pressure": float,
    "timing_advance": float
  },
  "vehicle_context": {
    "make": "string",
    "model": "string",
    "year": int,
    "engine": "string",
    "mileage": int
  },
  "repair_summary": "string",  // Concise summary of repair steps and solution (50-200 words)
  "outcome": "success|failure|partial|unknown"
}
```

#### Optional but Valuable Fields
- `symptoms`: User-reported symptoms
- `diagnostic_steps`: Steps taken before repair
- `parts_used`: Actual parts replaced
- `cost`: Repair cost (if mentioned)
- `time_taken`: Actual repair time
- `follow_up`: Follow-up information
- `related_codes`: Other codes that appeared together
- `source_type`: "forum|video|documentation|tsb"
- `confidence_score`: Quality/confidence of the data

### 4. Data Quality Criteria

#### High Priority (Must Have)
- ✅ Valid fault code format (P-codes, manufacturer codes)
- ✅ At least basic vehicle context (make/model/year)
- ✅ Repair summary (description of repair steps/solution)
- ✅ Outcome or resolution status

#### Medium Priority (Should Have)
- ⚠️ OBD-II sensor data (at least 3-5 parameters)
- ⚠️ Detailed repair summary with multiple steps
- ⚠️ Multiple fault codes (code combinations)

#### Low Priority (Nice to Have)
- ℹ️ Detailed symptoms
- ℹ️ Part numbers
- ℹ️ Cost information
- ℹ️ Follow-up confirmation

### 5. Collection Strategy

#### Phase 1: Broad Collection (Week 1-2)
- Collect 5,000-10,000 raw data points
- Focus on high-traffic forums and documentation sites
- Prioritize BMW and common European vehicles (system's primary focus)
- Include all fault code types (P0xxx, P1xxx, P2xxx, P3xxx, manufacturer codes)

#### Phase 2: Targeted Collection (Week 3-4)
- Focus on cases with OBD-II data
- Collect code combinations (multiple faults together)
- Target specific high-frequency fault codes (P0300, P0420, P0171, etc.)
- Collect vehicle-specific variations

#### Phase 3: Quality Enhancement (Week 5-6)
- Verify and clean collected data
- Fill in missing OBD-II data where possible
- Cross-reference multiple sources for validation
- Remove duplicates and low-quality entries

### 6. Specific Search Queries

#### For Forums & Communities
```
- "P0300" + "BMW" + "fix"
- "fault code" + "diagnosis" + "repair"
- "OBD-II" + "reading" + "solution"
- "[fault_code]" + "335i" + "N55"
- "check engine light" + "diagnosis" + "steps"
```

#### For Documentation Sites
```
- Fault code definitions
- Diagnostic procedures
- Repair guides
- Technical service bulletins
- OBD-II code explanations
```

#### For Video Content
```
- "[fault_code] diagnosis"
- "BMW fault code repair"
- "OBD-II scanner reading"
- "check engine light fix"
```

### 7. Data Validation Rules

#### Fault Code Validation
- Must match pattern: `P[0-9][0-9][0-9][0-9]` or manufacturer-specific format
- Common formats: P0300, P0420, P0171, BMW codes (e.g., 2A87, 2F2E)

#### OBD-II Data Validation
- Numeric values only
- Reasonable ranges:
  - engine_rpm: 0-8000
  - coolant_temp: -40 to 150 (Celsius)
  - throttle_position: 0-100 (%)
  - maf_sensor: 0-1000 (g/s)
  - fuel_trim: -100 to 100 (%)

#### Vehicle Context Validation
- Make: Must be valid manufacturer name
- Model: Must be valid model name
- Year: 1980-2024 (reasonable range)
- Engine: Valid engine code or description

### 8. Output Format

#### File Structure
```
mist_training_data/
├── raw_data/
│   ├── forums/
│   │   ├── reddit_*.jsonl
│   │   ├── bimmerforums_*.jsonl
│   │   └── ...
│   ├── documentation/
│   │   ├── obd_codes_*.jsonl
│   │   └── repair_guides_*.jsonl
│   ├── videos/
│   │   └── youtube_*.jsonl
│   └── tsbs/
│       └── tsb_*.jsonl
├── processed_data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── metadata/
    ├── collection_stats.json
    ├── source_urls.txt
    └── data_quality_report.json
```

#### JSONL Format (One JSON object per line)
```jsonl
{"fault_codes": ["P0300"], "obd_data": {...}, "vehicle_context": {...}, "repair_summary": "Replaced ignition coil on cylinder 1. Checked spark plug condition.", "outcome": "success", "source_url": "https://...", "timestamp": "2024-01-15T10:30:00Z"}
{"fault_codes": ["P0420"], "obd_data": {...}, "vehicle_context": {...}, "repair_summary": "Replaced oxygen sensor bank 1. Cleared codes and verified no return.", "outcome": "success", "source_url": "https://...", "timestamp": "2024-01-15T10:31:00Z"}
```

### 9. Ethical & Legal Considerations

#### Respect Website Terms
- Check robots.txt before scraping
- Respect rate limits (max 1 request per 2 seconds)
- Use proper User-Agent headers
- Don't overload servers

#### Data Privacy
- Remove personal information (names, addresses, phone numbers)
- Anonymize user handles/usernames
- Don't collect private messages or sensitive data

#### Attribution
- Always include source URLs
- Preserve original context when possible
- Note data collection date

### 10. Quality Metrics

#### Collection Targets
- **Minimum**: 5,000 high-quality records
- **Target**: 10,000+ records
- **Ideal**: 20,000+ records

#### Distribution Targets
- At least 500 unique fault codes
- Coverage of all major fault categories (P0xxx, P1xxx, P2xxx, P3xxx)
- At least 30% of records with OBD-II data
- At least 50% with vehicle context
- At least 70% with repair summaries
- At least 40% with outcome information

#### Data Quality Score
Calculate for each record:
```
quality_score = (
  (has_fault_code ? 0.3 : 0) +
  (has_vehicle_context ? 0.2 : 0) +
  (has_repair_summary ? 0.3 : 0) +
  (has_obd_data ? 0.15 : 0) +
  (has_outcome ? 0.05 : 0)
)
```
Minimum quality score: 0.6 (60%)

### 11. Special Instructions

#### For BMW-Specific Data (High Priority)
- Focus on BMW fault codes (2A87, 2F2E, etc.)
- Collect ISTA-specific diagnostic procedures
- Include BMW model codes (E90, F30, etc.)
- Note engine codes (N54, N55, B58, etc.)

#### For Code Combinations
- When multiple fault codes appear together, create separate records for:
  1. Individual codes
  2. Code combinations (as single record)
- This helps the system learn code relationships

#### For OBD-II Data
- If OBD-II data is not available, try to infer from text descriptions
- Look for phrases like "RPM was 2500", "coolant temp 95°C"
- Extract numeric values from text when possible

### 12. Progress Reporting

Report every 1,000 records collected:
- Total records collected
- Records by source type
- Records by fault code category
- Average quality score
- Data distribution statistics
- Any issues or blockers

### 13. Success Criteria

The collection is successful when:
- ✅ At least 5,000 records with quality_score >= 0.6
- ✅ Coverage of at least 500 unique fault codes
- ✅ At least 30% of records include OBD-II data
- ✅ Data spans multiple vehicle makes/models
- ✅ Includes both successful and failed repair outcomes
- ✅ Data is properly formatted and validated

## Final Instructions

1. **Extract repair summaries**: From scraped content, extract concise summaries of repair steps, diagnostic procedures, and solutions (50-200 words)
2. **Do NOT match repair guides during scraping**: This will be done in a separate processing step using the vector database
3. **Start with high-value sources**: Forums and documentation sites
4. **Prioritize quality over quantity**: Better to have 5,000 good records than 20,000 poor ones
5. **Validate as you go**: Don't wait until the end to validate data
6. **Document everything**: Keep track of sources, collection dates, and any issues
7. **Respect rate limits**: Be a good web citizen
8. **Focus on BMW/European vehicles**: This is the system's primary use case
9. **Collect OBD-II data when available**: This is critical for multi-modal training
10. **Write clear repair summaries**: Focus on what was done, not matching to guides

**Post-Processing:**
After scraping, use the `match_repair_guides.py` script to match repair summaries to repair guides in the vector database. This separation allows for:
- ✅ Iterative improvement of matching logic without re-scraping
- ✅ Batch processing of all collected data
- ✅ Consistent matching using the same semantic search pipeline as production
- ✅ Better handling of edge cases and fallback strategies

Begin collection immediately and report progress every 1,000 records.
