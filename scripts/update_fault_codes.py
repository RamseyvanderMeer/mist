import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# Load env
load_dotenv(Path('.').resolve() / '.env')

def get_codes():
    engine = create_engine(os.environ['DATABASE_URL'])
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT DISTINCT code 
            FROM scraped_records, 
            jsonb_array_elements_text(CAST(fault_codes AS JSONB)) as code
        """))
        codes = sorted([row[0] for row in r])
    return codes

def update_config(codes):
    config_path = Path("scrapers/utils/forum_config.py")
    with open(config_path, "r") as f:
        content = f.read()
    
    # Split content to preserve imports and FORUM_CONFIGS
    # We want to replace FAULT_CODES_TO_SEARCH = [...]
    
    start_marker = "FAULT_CODES_TO_SEARCH = ["
    end_marker = "]"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Could not find FAULT_CODES_TO_SEARCH")
        return

    # Find the closing bracket for the list
    # We need to be careful not to match nested brackets if any (though unlikely here)
    # But easier: finding the next "FORUM_CONFIGS =" or end of file
    
    next_section = content.find("FORUM_CONFIGS =", start_idx)
    if next_section == -1:
        print("Could not find FORUM_CONFIGS")
        return
        
    # Find the last ']' before FORUM_CONFIGS
    end_idx = content.rfind("]", start_idx, next_section)
    if end_idx == -1:
        print("Could not find closing bracket")
        return
        
    # Construct new list content
    new_list = "FAULT_CODES_TO_SEARCH = [\n"
    
    # Group by 8 for readability
    for i in range(0, len(codes), 8):
        chunk = codes[i:i+8]
        quoted = [f'"{c}"' for c in chunk]
        new_list += "    " + ", ".join(quoted) + ",\n"
        
    new_list += "]"
    
    new_content = content[:start_idx] + new_list + content[end_idx+1:]
    
    with open(config_path, "w") as f:
        f.write(new_content)
    
    print(f"Updated {config_path} with {len(codes)} codes")

if __name__ == "__main__":
    codes = get_codes()
    update_config(codes)
