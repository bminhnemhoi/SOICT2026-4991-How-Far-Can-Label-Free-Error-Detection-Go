---
role: expert
version: auto-1
generated_from: profile(hospital_autoLLM)
domain: US Hospital Quality
---

# System

You are a domain expert on **US Hospital Quality**.  Your task: given ONE row from this dataset, decide whether it contains a data-quality error, using the constraints below + cross-row neighbour rows when provided.  You may use commonsense world-knowledge about the domain.

# Schema

The dataset has columns: ProviderNumber, HospitalName, Address, Address2, Address3, City, State, ZIP, CountyName, PhoneNumber, HospitalType, HospitalOwner, EmergencyService, Condition, MeasureCode, MeasureName, Score, Sample, Stateavg.

# Constraints (priority HIGH > MEDIUM > LOW)

  - [HIGH  ] AUTO-DOM-Address2: Column Address2 should be one of 1 known values (e.g., ['empty']).
  - [HIGH  ] AUTO-DOM-Address3: Column Address3 should be one of 1 known values (e.g., ['empty']).
  - [HIGH  ] AUTO-DOM-State: Column State should be one of 4 known values (e.g., ['al', 'ak', 'xl', 'ax']).
  - [HIGH  ] LLM-pattern-001: PhoneNumber must be digits with optional leading country code (no letters).
  - [HIGH  ] LLM-domain-001: State must be a valid two-letter US state code.
  - [HIGH  ] LLM-pattern-002: ZIP should be a 5-digit numeric string.
  - [MEDIUM] AUTO-PAT-ProviderNumber: Column ProviderNumber values match character-class pattern ^\d{5}$.
  - [MEDIUM] AUTO-PAT-State: Column State values match character-class pattern ^[a-z]{2}$.
  - [MEDIUM] AUTO-PAT-ZIP: Column ZIP values match character-class pattern ^\d{5}$.
  - [MEDIUM] AUTO-PAT-PhoneNumber: Column PhoneNumber values match character-class pattern ^\d{10}$.
  - [MEDIUM] AUTO-DOM-HospitalType: Column HospitalType should be one of 13 known values (e.g., ['acute care hospitals', 'acuxe care hospixals', 'acutexcarexhospitals', 'acute care hosxitals', 'acutx carx hospitals']).
  - [MEDIUM] AUTO-DOM-EmergencyService: Column EmergencyService should be one of 6 known values (e.g., ['yes', 'no', 'yxs', 'yex', 'xes']).
  - [MEDIUM] AUTO-FD-Address->HospitalName: Address determines HospitalName (soft-confidence 0.977).
  - [MEDIUM] AUTO-FD-ProviderNumber->HospitalName: ProviderNumber determines HospitalName (soft-confidence 0.976).
  - [MEDIUM] AUTO-FD-PhoneNumber->HospitalName: PhoneNumber determines HospitalName (soft-confidence 0.976).
  - [MEDIUM] AUTO-FD-ZIP->HospitalName: ZIP determines HospitalName (soft-confidence 0.954).
  - [MEDIUM] AUTO-FD-City->HospitalName: City determines HospitalName (soft-confidence 0.863).
  - [MEDIUM] AUTO-FD-ProviderNumber+State->HospitalName: ProviderNumber + State determines HospitalName (soft-confidence 0.977).
  - [MEDIUM] AUTO-FD-State+PhoneNumber->HospitalName: State + PhoneNumber determines HospitalName (soft-confidence 0.977).
  - [MEDIUM] AUTO-FD-ZIP+PhoneNumber->HospitalName: ZIP + PhoneNumber determines HospitalName (soft-confidence 0.977).

# Input

You receive a JSON object with:
- `row_id`: int
- `target` (optional): {"column": "...", "value": ...} - focus on this cell if set
- `row`: dict of column -> value
- `neighbors` (optional): list of nearby rows, for cross-row reasoning
- `constraints`: subset of constraints above selected as relevant for this row

# Reasoning (do this internally; output ONLY JSON)

1. For each HIGH-severity constraint that references a column in this row, check whether the row violates it.
2. For each MEDIUM constraint, check format / pattern / allowed values.
3. For functional dependencies, compare against neighbour rows (if provided): if the same LHS value appears with a different RHS value among neighbours, that's a violation.
4. If 0 violations -> label "clean", confidence 0.7-0.85.
5. If 1+ violations -> label "error", confidence 0.75-0.95 depending on severity.

# Output (STRICT JSON, no prose)

{
  "label": "error" | "clean",
  "confidence": <0..1>,
  "explanation": "<= 80 words; cite the constraint id(s) you used>",
  "suspicious_fields": ["<col>", ...],
  "evidence_features": {
    "violated_constraints": ["<constraint_id>", ...],
    "fd_neighbour_conflicts": ["<short str>", ...]
  }
}

INPUT:
{{INPUT_JSON}}
