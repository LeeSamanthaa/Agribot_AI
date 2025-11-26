# coding=utf-8
"""
run_chatbot.py 
Author: Samantha Lee
This script initializes and runs the CropHealthChatbotAgentV6
in a terminal for testing.
"""

import os
import sys
from pathlib import Path
import traceback

# --- CRITICAL FIX: INJECT CORRECT PROJECT ROOT ---
# This line ensures that the directory containing run_chatbot.py 
# (e.g., 'C:\Users\SCLee\OneDrive\Projects\Crop Health') is set as the root,
# allowing Python to find 'ai_agent.src.agents'.
PROJECT_ROOT = Path(__file__).resolve().parent 
if str(PROJECT_ROOT) not in sys.path:
    # Insert the root path so Python can find packages like 'ai_agent.src.agents'
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"--- [Debug] Path set to Project Root: {PROJECT_ROOT}")
# --------------------------------------------------------------------
    
print("--- [Debug] Importing agent (this may take a while)...")

try:
    # This import now works because the project root is on the path
    from ai_agent.src.agents.crop_health_chatbot_agentV6 import CropHealthChatbotAgentV6
except ImportError as e:
    print(f"--- [Debug] IMPORT FAILED! ---")
    print(f"Error: {e}")
    print("Ensure the PROJECT_ROOT path is correct and your virtual environment is active.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


print("--- [Debug] Agent imported successfully.")

# --- Main Execution Block ---
if __name__ == "__main__":
    
    DATE = "2025-10-22" 
    # Use a default field for initial load
    FIELD_ID = "vtx|Fejer|rapeseed|0x0|+236033+262315" 

    print(f"\nInitializing chatbot for Field: {FIELD_ID} on Date: {DATE}...")
    
    try:
        # This init is slow (loads model, scaler, and data)
        chatbot = CropHealthChatbotAgentV6(date=DATE, field_id=FIELD_ID)
    except Exception as e:
        print(f"\n--- CRITICAL ERROR initializing chatbot ---")
        print(f"Failed to load models or data: {e}")
        print("Please check data and model paths in 'crop_health_model_agentV6.py' relative to the PROJECT_ROOT.")
        sys.exit(1)

    
    all_locs = chatbot.all_locations_list
    ex1 = all_locs[0] if len(all_locs) > 0 else "Vas"
    ex2 = all_locs[1] if len(all_locs) > 1 else "Budapest"
    
    print("\n" + "=" * 60)
    print("CropLogic AI Assistant (GRU Model V6.0 - Terminal Test)")
    print("=" * 60)
    
    print(f"\nHello! Monitoring the {chatbot.crop_type} field in {chatbot.location_name}.")
    print(f"(Field ID: {chatbot.predictor.field_id})")
    print(f"Data is current as of: {chatbot.predictor.date.strftime('%Y-%m-%d')}")
    
    print("\nThis agent uses a 60-day lookback model to provide deep insights.")
    print("You can ask for:")
    print(f"  • 'Give me a 7-day forecast for {ex1}'")
    print(f"  • 'forecast {ex2} sunflower'")
    print("  • 'What's the current crop status?'")
    print("  • 'What is NDVI?'")
    
    print("\nType 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nGoodbye!")
                break
                
            response_string = chatbot.handle_input(user_input)
            print(f"\n{response_string}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            traceback.print_exc()
            print("Please try again.\n")