import os
import json
import requests
import datetime
import dateutil.parser
from google import genai
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SETUP (NEW SDK) ---

# Load environment variables from .env file
load_dotenv()
# The new SDK prefers the 'GEMINI_API_KEY' variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file")

# Initialize the new, centralized client
# The client will automatically find and use the GEMINI_API_KEY
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure your GEMINI_API_KEY is correct.")
    exit()

# Constants
SALES_API_ENDPOINT = "https://sandbox.mkonnekt.net/ch-portal/api/v1/orders/recent"
CACHE_DURATION = datetime.timedelta(minutes=5)

# In-memory cache
_api_cache = {
    "data": None,
    "timestamp": None,
}

# --- 2. API CLIENT (with Caching & Validation) ---

def get_sales_data():
    #fetches sales data from the API with caching and response validation.
    global _api_cache

    now = datetime.datetime.now()

    # Check if cache is valid
    if _api_cache["data"] and _api_cache["timestamp"]:
        if now - _api_cache["timestamp"] < CACHE_DURATION:
            print("[INFO] Using cached API data.")
            return _api_cache["data"]

    print("[INFO] Fetching new data from Sales API...")
    try:
        response = requests.get(SALES_API_ENDPOINT)
        # Raise an error for bad responses (4xx or 5xx)
        response.raise_for_status() 
        
        data = response.json()
        orders_list = None
        
        # Check if the response is a dictionary (as hypothesized)
        if isinstance(data, dict):
            # Try to find the list inside common keys
            if 'data' in data and isinstance(data.get('data'), list):
                print("[INFO] Found order list inside 'data' key.")
                orders_list = data['data']
            elif 'results' in data and isinstance(data.get('results'), list):
                print("[INFO] Found order list inside 'results' key.")
                orders_list = data['results']
            elif 'orders' in data and isinstance(data.get('orders'), list):
                print("[INFO] Found order list inside 'orders' key.")
                orders_list = data['orders']
            else:
                # The structure is a dict, but we don't know the key.
                print(f"[ERROR] API returned a dictionary, but could not find the order list: {data}")
                return None
        elif isinstance(data, list):
            # The API returned a list directly, as originally expected.
            print("[INFO] API returned a list directly.")
            orders_list = data
        else:
            # The API returned something else (e.g., a string)
            print(f"[ERROR] API returned an unexpected response (not a list or dict): {data}")
            return None

        # Update cache with the *actual list*
        _api_cache["data"] = orders_list
        _api_cache["timestamp"] = now
        print(f"[INFO] API data fetched and cached ({len(orders_list)} orders).")
        return orders_list

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode API response as JSON.")
        return None


# --- 3. SMART DATE PARSING (NEW SDK) ---

def get_date_range_from_llm(client: genai.Client, user_query: str) -> (datetime.date, datetime.date):
    print(f"[INFO] Parsing date for query: '{user_query}'")
    
    today = datetime.date.today()
    prompt = f"""
    You are a date parsing assistant. Today's date is {today.isoformat()}.
    Analyze the user's query and determine the start date and end date (inclusive) for their request.

    User Query: "{user_query}"

    Respond ONLY with a JSON object in the format:
    {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}
    
    Examples:
    - Query "yesterday": {{"start_date": "{today - datetime.timedelta(days=1)}", "end_date": "{today - datetime.timedelta(days=1)}"}}
    - Query "today": {{"start_date": "{today}", "end_date": "{today}"}}
    - Query "this month": {{"start_date": "{today.replace(day=1)}", "end_date": "{today}"}}
    - Query "last week" (assume Mon-Sun): {{"start_date": "2025-10-27", "end_date": "2025-11-02"}}
    - Query "how much revenue?": {{"start_date": "{today}", "end_date": "{today}"}} (Defaults to today)
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',  # Use the model from your template
            contents=prompt
        )
        
        # Clean up the response to get pure JSON
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        dates = json.loads(json_str)
        
        start_date = dateutil.parser.isoparse(dates["start_date"]).date()
        end_date = dateutil.parser.isoparse(dates["end_date"]).date()
        
        print(f"[INFO] Date range parsed: {start_date} to {end_date}")
        return start_date, end_date
        
    except Exception as e:
        print(f"[ERROR] Failed to parse date with LLM: {e}. Defaulting to today.")
        today = datetime.date.today()
        return today, today


# --- 4. DATA FILTERING ---
def filter_orders_by_date(all_orders: list, start_date: datetime.date, end_date: datetime.date) -> list:
    # Filters orders to only those completed within the date range.
    print(f"[INFO] Filtering orders from {start_date} to {end_date}...")
    filtered_orders = []
    
    # Create datetime objects for comparison (start of day, end of day)
    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt = datetime.datetime.combine(end_date, datetime.time.max)

    if not all_orders:
        return []

    for order in all_orders:
        if order.get("state") == "locked":
            try:
                # Parse the ISO 8601 timestamp
                created_time_str = order.get("createdTime")
                if not created_time_str:
                    continue
                    
                # The API timestamp has no timezone, so we parse it as naive
                created_dt = dateutil.parser.isoparse(created_time_str)
                
                # Check if it's in the date range
                if start_dt <= created_dt <= end_dt:
                    filtered_orders.append(order)
            except Exception as e:
                print(f"[WARN] Could not parse order {order.get('orderId')}: {e}")
                
    print(f"[INFO] Found {len(filtered_orders)} completed orders in date range.")
    return filtered_orders


# --- 5. LLM ANALYSIS (NEW SDK) ---

def get_analysis_from_gemini(client: genai.Client, user_query: str, orders: list):
    print("[INFO] Sending data to Gemini for analysis...")

    # This is the "System Prompt" that instructs the model
    system_instruction = """
    You are a friendly and expert sales analysis assistant.
    You will be given a user's question and a list of sales orders in JSON format.
    Your task is to analyze the JSON data to answer the user's question.

    **CRITICAL INSTRUCTIONS:**
    1.  All currency values in the JSON (like 'total' and 'lineItems[].price') are in **CENTS**.
    2.  When you present your answer, **ALWAYS** convert these cents to dollars (e.g., 906 cents is $9.06).
    3.  Only use the provided JSON data for your analysis.
    4.  Answer in a clear, natural, and friendly tone. Use markdown for formatting (like lists).
    5.  If the question is about 'best-selling items', analyze the 'lineItems' across all orders.
    6.  If the JSON list is empty, inform the user you found no sales data for that period.
    7.  The 'state' field 'locked' means the order is completed. You will only receive locked orders.
    """

    # Create the user prompt for the model
    prompt_to_llm = f"""
    User Question: "{user_query}"

    Here is the sales data for the relevant period. Please analyze it:
    {json.dumps(orders, indent=2)}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[system_instruction, prompt_to_llm] # Pass both as a list
        )
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini analysis failed: {e}")
        return "I'm sorry, I encountered an error while analyzing the sales data."

# --- 6. MAIN APPLICATION LOOP (NEW SDK) ---

def main():
    # Main function to run the CLI agent.
    print("\n--- 🤖 Welcome to the Sales Insight Agent ---")
    print("Ask me about your sales! (e.g., 'What were our best-selling items yesterday?')")
    print("Type 'exit' to quit.\n")
    
    # The client is already initialized above
    global client

    while True:
        try:
            user_query = input("> ")
            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            print("[INFO] Processing your request...")
            
            # 1. Get all sales data (from cache or API)
            all_orders = get_sales_data()
            if all_orders is None:
                # This now catches both API failures and bad data responses
                print("I'm sorry, I couldn't retrieve valid sales data.")
                continue

            # 2. Parse date range from query (pass the client)
            start_date, end_date = get_date_range_from_llm(client, user_query)

            # 3. Filter orders based on date range and state
            filtered_orders = filter_orders_by_date(all_orders, start_date, end_date)

            # 4. Get final analysis from LLM (pass the client)
            analysis = get_analysis_from_gemini(client, user_query, filtered_orders)

            # 5. Present result
            print(f"\nAnalysis for '{user_query}' ({start_date} to {end_date}):")
            print("---")
            print("Analysis Result:\n")
            print(len(analysis) > 0 and analysis or "No analysis available.")
            print("---\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")
            print("Please try your query again.\n")

if __name__ == "__main__":
    main()