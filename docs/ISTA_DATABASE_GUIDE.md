# ISTA Database Architecture & Agent Instructions

This guide explains the internal structure of the BMW ISTA databases, how to access them, and specifically how to perform a "Process Analysis" to trace the linkage between repair procedures and their preliminary tasks.

---

## 1. Database Fundamentals

### The Files
The core intelligence of ISTA resides in SQLite databases. We work with the decrypted versions.

1.  **`DiagDocDb_Decrypted.sqlite`** (The "Brain")
    *   **Purpose:** Contains the relational map of vehicles, components, faults, and document metadata.
    *   **Key Tables:** `XEP_INFOOBJECTS`, `XEP_REFDOCUMENTS`, `XEP_VEHICLES`, `VINRANGES`, `XEP_CHARACTERISTICS`.
    *   **Access:** Standard SQLite3 (after decryption).

2.  **`xmlvalueprimitive_ENGB.sqlite`** (The "Library")
    *   **Purpose:** Contains the actual *text content* (XML) of the documents.
    *   **Key Table:** `xmlvalueprimitive` (and its FTS index `fts`).
    *   **Linking:** Linked via `CONTENTID` (from `XEP_INFOSEGMENTS`) or implicitly by Title/ID.

3.  **`streamdataprimitive_OTHER.sqlite`** (The "Gallery")
    *   **Purpose:** Contains binary blobs (images, graphics).
    *   **Key Table:** `streamdataprimitive`.
    *   **Linking:** Linked via `CONTENTID`.

### Connection
*   **Engine:** SQLite3.
*   **Password:** The original DBs are encrypted (SQLCipher).
    *   **Password:** `6505EFBDC3E5F324` (Raw key) or via `pysqlcipher3` PRAGMA.
    *   *Note:* We are using `DiagDocDb_Decrypted.sqlite`, which has **no password**. You can use standard `sqlite3` library.

---

## 2. The Document Hierarchy (Process Analysis)

To understand how a "Main Procedure" knows about its "Preliminary Tasks", you must understand the ISTA structural nodes.

### The Chain of Command
1.  **Diagnosis Object (`XEP_DIAGNOSISOBJECTS`)**
    *   Represents a "Repair Node" or "Service Function" in the functional tree.
    *   This is the *parent* entity that holds validity rules and structural links.
    *   **Key Column:** `CONTROLID`.

2.  **Info Object (`XEP_INFOOBJECTS`)**
    *   Represents the actual document metadata (Title, DocNumber).
    *   Linked to Diagnosis Object via `XEP_REFINFOOBJECTS`.
    *   **Key Column:** `CONTROLID` (Unique to the InfoObject).

3.  **Info Segment (`XEP_INFOSEGMENTS`)**
    *   Represents a chunk of XML content within an InfoObject.
    *   **Key Column:** `CONTROLID` (Unique to the Segment).

### Linkage Logic (How `Ref: h3` works)

When a document says `(Ref: h3)`, it is a **relative link**. The resolution logic is hierarchical:

1.  **Segment Level (Most Specific)**
    *   Check `XEP_REFDOCUMENTS` where `INFOOBJECTCONTROLID` = **Segment ControlID**.
    *   Match `LINKID` (e.g., "h3").
    *   *Result:* Target `ID` (points to `XEP_INFOOBJECTS.ID`).

2.  **Document Level**
    *   Check `XEP_REFDOCUMENTS` where `INFOOBJECTCONTROLID` = **InfoObject ControlID**.
    *   Match `LINKID`.

3.  **Parent/Diagnosis Level (The "Missing Link")**
    *   If the above fail, the link might be inherited from the parent Diagnosis Object.
    *   Find Parent: `SELECT ID FROM XEP_REFINFOOBJECTS WHERE INFOOBJECTID = [MyDocID]`.
    *   Get Parent ControlID from `XEP_DIAGNOSISOBJECTS`.
    *   Check `XEP_REFDOCUMENTS` where `INFOOBJECTCONTROLID` = **Parent ControlID**.

---

## 3. Agent Task: Running a Process Analysis

**Objective:** Map the full execution tree for a given repair.

### Step-by-Step Algorithm

**Input:** A Document Number (e.g., `1124571`) or Title.

1.  **Resolve the Root InfoObject:**
    ```sql
    SELECT ID, CONTROLID, TITLE_ENGB FROM XEP_INFOOBJECTS WHERE DOCNUMBER = '1124571';
    ```

2.  **Fetch Content & Parse XML:**
    *   Get XML from `xmlvalueprimitive_ENGB` using the Title (FTS match).
    *   Parse XML for `<hotspot linkid="...">` or `<reference linkid="...">`.

3.  **Resolve References (The Core Loop):**
    *   For every `linkid` found in the XML:
        *   Query `XEP_REFDOCUMENTS`:
            ```sql
            SELECT ID FROM XEP_REFDOCUMENTS 
            WHERE INFOOBJECTCONTROLID = [Root_ControlID] 
            AND LINKID = [linkid];
            ```
        *   **Target ID:** The result is the `ID` of the *next* `XEP_INFOOBJECT`.

4.  **Recurse:**
    *   Take the Target ID.
    *   Fetch *its* content.
    *   Repeat Step 2 & 3.

### Handling Broken Links (Fallback)
If `XEP_REFDOCUMENTS` returns nothing (common for some legacy docs):
1.  **Text Heuristic:** Extract the text *preceding* the link in the XML.
    *   *Example:* "Removing oil sump `<hotspot linkid="h3">...`" -> Text: "Removing oil sump".
2.  **Fuzzy Search:** Perform a broad search in `XEP_INFOOBJECTS` for that text.
    *   *Filtering:* Use the **Main Group** of the parent doc (first 2 digits of DocNumber, e.g., "11") to filter the fallback results. This ensures you get "Engine Oil Sump" (Group 11) and not "Transmission Oil Sump" (Group 24).

---

## 4. Code Snippets for Agents

### Reference Resolution Query
```python
def resolve_ref(db, control_id, link_id):
    # Case-insensitive check is crucial!
    # LINKID in DB might be 'H3', query is 'h3'.
    cursor.execute("SELECT ID, LINKID FROM XEP_REFDOCUMENTS WHERE INFOOBJECTCONTROLID = ?", (control_id,))
    for row in cursor.fetchall():
        target_id, db_links = row
        if link_id.upper() in db_links.upper().split(','):
            return target_id
    return None
```

### Vehicle Context Filtering (Crucial)
When searching, you **must** filter by the vehicle's Engine/Chassis to avoid getting manuals for the wrong car.
*   **Engine Code:** S55 (M3) vs N55 (335i).
*   **Logic:** If a document title contains `(N55)` but your profile is `S55`, **discard it**.
*   **Look for:** `XEP_CHARACTERISTICS` matches for the vehicle model.

---

## 5. Known Pitfalls
*   **Unicode:** Titles and content often contain non-standard hyphens (`\u2010`). Sanitize text before processing.
*   **Terminology:** ISTA uses "Cylinder Head Cover", users say "Valve Cover". Use a synonym map.
*   **Structure:** "Preliminary Tasks" are often hidden inside `<INCLUDE_PROCESS>` tags. Your XML parser **must** be recursive.
