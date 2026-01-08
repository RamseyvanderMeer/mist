# BMW ISTA Database Access Guide
## How to Query the Database Without ISTA

Based on reverse engineering the deobfuscated DLLs, here's exactly how to access the BMW ISTA database structures.

---

## Database Connection

### The Right Way (System.Data.SQLite)

BMW ISTA uses **System.Data.SQLite** (NOT SQLCipher, NOT standard SQLite).

```csharp
using System.Data.SQLite;

string connectionString = "Data Source=D:\\apps\\ISTA\\SQLiteDBs\\DiagDocDb.sqlite;Password=6505EFBDC3E5F324;Version=3;";

using (var conn = new SQLiteConnection(connectionString))
{
    conn.Open();
    // Execute queries...
}
```

### Important Notes

1. **System.Data.SQLite** is NOT standard SQLite
2. The database uses **custom AES-256 encryption** built into System.Data.SQLite
3. SQLCipher and standard `sqlite3` command-line tools **WILL NOT WORK**
4. You **MUST** use System.Data.SQLite library to access the database

### Why Other Tools Fail

- `sqlite3`: Not aware of System.Data.SQLite encryption
- `sqlcipher`: Uses different encryption format (SQLCipher vs System.Data.SQLite)
- Python `sqlite3`: Standard SQLite, no encryption support
- Python `pysqlcipher3`: Wrong encryption format

---

## Database Tables

### Core Tables (from RheingoldDatabaseSQLiteConnector_deobfuscated.dll)

#### Vehicle Identification
- `VINRANGES` - VIN range to TYPSCHLUESSEL mapping (5.4M records)
- `XEP_TYPEKEY_MAPPING` - TYPSCHLUESSEL to development code mapping
- `XEP_VEHICLES` - Vehicle type information
- `XEP_VEHICLEPART` - Vehicle parts
- `XEP_VEHICLESTATE` - Vehicle state
- `XEP_VEHICLEADAPTER` - Vehicle adapters

#### Equipment & Configuration
- `XEP_EQUIPMENT` - SA codes (Sonderausstattung - special equipment)
- `XEP_SALAPAS` - SA code descriptions

#### ECU & Diagnostics
- `XEP_ECUVARIANTS` - ECU variants by vehicle
- `XEP_ECUGROUPS` - ECU groupings
- `XEP_ECUCLIQUES` - ECU cliques (groupings)
- `XEP_ECUFUNCSTRUCTURES` - ECU function structures
- `XEP_ECUGROUPFUNCTIONS` - ECU group functions
- `XEP_ECUVARFUNCTIONS` - ECU variant functions
- `XEP_ECUFIXEDFUNCTIONS` - ECU fixed functions
- `XEP_ECUSPECIALFUNCTION` - ECU special functions
- `XEP_ECUPROGRAMMINGVARIANT` - ECU programming variants
- `XEP_ECUJOBSEX` - Extended ECU jobs
- `XEP_ECUPARAMETERSEX` - Extended ECU parameters
- `XEP_ECURESULTSEX` - Extended ECU results
- `XEP_ECUREPS` - ECU representatives

#### Fault Codes
- `XEP_FAULTCODE` - Fault/error codes
- `XEP_FAULTLABELS` - Fault descriptions
- `XEP_COMBINEDFAULTS` - Combined fault information
- `XEP_COMBIFAULTLABELS` - Combined fault labels
- `XEP_FAULTCLASSES` - Fault classifications
- `XEP_FAULTMODELABELS` - Fault mode labels
- `XEP_VIRTUALFAULTCODES` - Virtual fault codes
- `XEP_VIRTUALFAULTLABELS` - Virtual fault labels

#### Diagnostic Procedures
- `XEP_DIAGNOSISOBJECTSEX` - Extended diagnosis objects
- `XEP_DIAGNOSISOBJECTS_TITLE` - Diagnosis object titles
- `XEP_REFDIAGOBJECTS` - Reference diagnosis objects
- `XEP_REFDIAGNOSISTREE` - Diagnosis tree references
- `XEP_QUERYOBJECTSEX` - Extended query objects
- `XEP_REF_DIAGCODE_ECU` - Diagnostic code to ECU reference

#### Other Tables
- `XEP_NODECLASSES` - Node classes
- `XEP_CHARACTERISTICROOTS` - Characteristic roots
- `XEP_CHARACTERISTICS` - Characteristics
- `XEP_ENVCONDSLABELS` - Environment condition labels
- `XEP_STATEVALUES` - State values
- `XEP_IOCONTENTS` - I/O contents
- `XEP_REFCONTENTS` - Reference contents
- `XEP_REFINFOOBJECTS` - Reference info objects
- `XEP_REFECUCLIQUES` - Reference ECU cliques
- `XEP_REFECUFUNCSTRUCTS` - Reference ECU function structures
- `XEP_REFECUVARIANTS` - Reference ECU variants
- `XEP_SPTEXTITEMS` - Special text items
- `XEP_CCMESSAGE` - CC messages
- `XEP_SWIACTION` - SWI actions
- `XEP_SWIREGISTER` - SWI register
- `XEP_SWIACTIVATIONCODE_IBAC` - SWI activation codes (IBAC)
- `XEP_SWIACTIVATIONCODE_SWT` - SWI activation codes (SWT)
- `XEP_WSCONFIGFILTERS` - Workshop config filters
- `XEP_VEHICLEVALUEOVERRIDE` - Vehicle value overrides

---

## Key Methods from DLL

### DatabaseProviderSQLite Class Methods

From `BMW.Rheingold.DatabaseProvider.SQLiteConnector.DatabaseProviderSQLite`:

#### VIN Lookup Methods
```csharp
// Get VINRANGES record by full VIN
VINRANGES GetVinRangesByVin(string vin)

// Get VINRANGES using VIN pattern (positions 4-7)
VINRANGES GetVinRangesByVin17(string vin17_4_7, string vin7, bool returnFirstEntryWithoutCheck)

// Get VINRANGES list by VIN17_4_7 pattern
List GetListofVinRangesByVin17_4_7(string vin17_4_7)
```

#### Type Key Methods
```csharp
// Get vehicle identification from TYPSCHLUESSEL
IList GetVehicleIdentByTypeKey(string typeKey, bool isAlpina)

// Get type key mapping
XEP_TYPEKEY_MAPPING GetTypeKeyMapping(decimal id)

// Get type keys for production period
IEnumerable<string> GetTypeKeysByYearAndMonthFromVinranges(string year, string month)

// Get distinct column values from VINRANGES by type key
GetDistinctColumnFromVinRangesByTypKey(string typeKey, string columnName)
```

---

## SQL Query Examples

### 1. VIN to TYPSCHLUESSEL

```sql
-- Convert VIN to internal format first
-- WMWSS5C5XEWN67289 -> WN67289 (plant code + serial)
-- Plant code = position 11 (index 10)
-- Serial = positions 12-17 (index 11-16)

SELECT
    TYPSCHLUESSEL,
    PRODUCTIONDATEYEAR,
    PRODUCTIONDATEMONTH,
    GEARBOX_TYPE,
    VIN17_4_7,
    RELEASESTATE
FROM VINRANGES
WHERE 'WN67289' BETWEEN VINBANDFROM AND VINBANDTO
LIMIT 1;
```

### 2. Find All Type Keys Matching Pattern

```sql
SELECT DISTINCT
    TYPSCHLUESSEL,
    COUNT(*) as count,
    MIN(PRODUCTIONDATEYEAR) as first_year,
    MAX(PRODUCTIONDATEYEAR) as last_year
FROM VINRANGES
WHERE TYPSCHLUESSEL LIKE 'SS%'
GROUP BY TYPSCHLUESSEL
ORDER BY TYPSCHLUESSEL;
```

### 3. Get Vehicle Information by Type Key

```sql
-- Check XEP_TYPEKEY_MAPPING for development code
SELECT *
FROM XEP_TYPEKEY_MAPPING
WHERE TYPEKEY = 'SS63';
```

### 4. Get Equipment Codes for Vehicle

```sql
SELECT *
FROM XEP_EQUIPMENT
WHERE TYPEKEY = 'SS63';
```

### 5. Get ECU Variants

```sql
SELECT *
FROM XEP_ECUVARIANTS
WHERE TYPEKEY = 'SS63';
```

### 6. Get Fault Codes

```sql
SELECT
    fc.CODE,
    fl.LABEL,
    fl.DESCRIPTION
FROM XEP_FAULTCODE fc
LEFT JOIN XEP_FAULTLABELS fl ON fc.ID = fl.FAULTCODE_ID
WHERE fc.TYPEKEY = 'SS63';
```

### 7. List All Tables

```sql
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name;
```

### 8. Explore Table Structure

```sql
PRAGMA table_info(VINRANGES);
```

---

## C# Example Code

```csharp
using System;
using System.Data.SQLite;

class BMWDatabaseAccess
{
    const string DB_PATH = @"D:\apps\ISTA\SQLiteDBs\DiagDocDb.sqlite";
    const string DB_PASSWORD = "6505EFBDC3E5F324";

    static void Main(string[] args)
    {
        string vin = "WMWSS5C5XEWN67289";

        // 1. Convert VIN to internal format
        string internalVin = VinToInternal(vin);
        Console.WriteLine($"Internal VIN: {internalVin}");

        // 2. Query database
        string connStr = $"Data Source={DB_PATH};Password={DB_PASSWORD};Version=3;";

        using (var conn = new SQLiteConnection(connStr))
        {
            conn.Open();

            string query = @"
                SELECT TYPSCHLUESSEL, PRODUCTIONDATEYEAR, PRODUCTIONDATEMONTH, GEARBOX_TYPE
                FROM VINRANGES
                WHERE @vin BETWEEN VINBANDFROM AND VINBANDTO
                LIMIT 1";

            using (var cmd = new SQLiteCommand(query, conn))
            {
                cmd.Parameters.AddWithValue("@vin", internalVin);

                using (var reader = cmd.ExecuteReader())
                {
                    if (reader.Read())
                    {
                        Console.WriteLine($"TYPSCHLUESSEL: {reader["TYPSCHLUESSEL"]}");
                        Console.WriteLine($"Production: {reader["PRODUCTIONDATEMONTH"]}/{reader["PRODUCTIONDATEYEAR"]}");
                        Console.WriteLine($"Gearbox: {reader["GEARBOX_TYPE"]}");
                    }
                }
            }
        }
    }

    static string VinToInternal(string vin17)
    {
        // BMW internal format: Plant code (pos 11) + Serial (pos 12-17)
        return vin17[10].ToString() + vin17.Substring(11, 6);
    }
}
```

---

## Python Access (Requires System.Data.SQLite)

**Note:** Pure Python cannot access System.Data.SQLite encrypted databases.

### Option 1: Use pythonnet

```python
import clr
clr.AddReference("System.Data.SQLite")
from System.Data.SQLite import SQLiteConnection

conn_str = "Data Source=D:\\apps\\ISTA\\SQLiteDBs\\DiagDocDb.sqlite;Password=6505EFBDC3E5F324;Version=3;"
conn = SQLiteConnection(conn_str)
conn.Open()

# Execute queries using .NET SQLite API
```

### Option 2: Create C# Wrapper DLL

Create a C# library that wraps database access, then call from Python.

---

## Common Issues

### Issue 1: "file is not a database"
**Cause:** Using wrong SQLite library (not System.Data.SQLite)
**Solution:** Use System.Data.SQLite, not standard SQLite or SQLCipher

### Issue 2: "incorrect format" (0x8007000B)
**Cause:** Architecture mismatch (x64 vs x86)
**Solution:** Compile with `/platform:x86` flag

### Issue 3: Cannot decrypt database
**Cause:** Wrong password or encryption library
**Solution:** Use password `6505EFBDC3E5F324` with System.Data.SQLite

---

## Tools Required

### Windows
- System.Data.SQLite.dll (from BMW ISTA installation)
- .NET Framework or .NET Core
- Visual Studio or `csc.exe` compiler

### macOS/Linux
- Mono (for .NET runtime)
- System.Data.SQLite.dll (cross-platform version)
- pythonnet (optional, for Python access)

---

## Next Steps

1. **Use C# with System.Data.SQLite** - This is the most reliable method
2. **Explore XEP_TYPEKEY_MAPPING** - This table likely contains TYPSCHLUESSEL to development code mappings
3. **Build mapping table** - Create a lookup table of TYPSCHLUESSEL -> Development Code
4. **Investigate GetVehicleIdentByTypeKey** - This DLL method is how ISTA gets development codes

---

## Database Files

- `DiagDocDb.sqlite` (6.04 GB) - Main diagnostic database
- `xmlvalueprimitive_ENGB.sqlite` (45 GB) - XML value primitives (English)
- `xmlvalueprimitive_DEDE.sqlite` (45 GB) - XML value primitives (German)
- `streamdataprimitive_OTHER.sqlite` (17 GB) - Stream data primitives

Only `DiagDocDb.sqlite` is needed for VIN identification.

---

## Password Generation

The password comes from BMW's PublicKeyToken:

```csharp
byte[] publicKeyToken = { 0x65, 0x05, 0xEF, 0xBD, 0xC3, 0xE5, 0xF3, 0x24 };
string password = "";
foreach (byte b in publicKeyToken)
{
    password += b.ToString("X2");
}
// Result: "6505EFBDC3E5F324"
```

This is the same PublicKeyToken used for all BMW ISTA assemblies.
