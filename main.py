import vertexai
from vertexai.preview import reasoning_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

import requests

PROJECT_ID = ""
LOCATION = ""
STAGING_BUCKET = ""
API_KEY = ""

def init_vertexai():
    # Initialize Vertex AI
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

def tavily_search_method(search_query: str = "where is sydney?"):
    """
    Retrieves results using Tavily search

    Uses the Tavily API (https://api.tavily.com/search) to obtain
    exchange rate data.

    Args:
        search_query: whatever you want to serach

    Returns:
        dict: {'query': 'What is the weather in Sydney?', 'follow_up_questions': None, 'answer': 'The current weather in Sydney is 12.3°C with clear skies. The wind is blowing at 11.2 kph from the SSW direction. The humidity level is at 88%, and there is no precipitation expected.', 'images': None, 'results': [{'title': 'Sydney, New South Wales, Australia Monthly Weather | AccuWeather', 'url': 'https://www.accuweather.com/en/au/sydney/22889/june-weather/22889', 'content': 'Get the monthly weather forecast for Sydney, New South Wales, Australia, including daily high/low, historical averages, to help you plan ahead.', 'score': 0.98987, 'raw_content': None}, {'title': 'Weather in Sydney', 'url': 'https://www.weatherapi.com/', 'content': "{'location': {'name': 'Sydney', 'region': 'New South Wales', 'country': 'Australia', 'lat': -33.88, 'lon': 151.22, 'tz_id': 'Australia/Sydney', 'localtime_epoch': 1719135568, 'localtime': '2024-06-23 19:39'}, 'current': {'last_updated_epoch': 1719135000, 'last_updated': '2024-06-23 19:30', 'temp_c': 12.3, 'temp_f': 54.1, 'is_day': 0, 'condition': {'text': 'Clear', 'icon': '//cdn.weatherapi.com/weather/64x64/night/113.png', 'code': 1000}, 'wind_mph': 6.9, 'wind_kph': 11.2, 'wind_degree': 210, 'wind_dir': 'SSW', 'pressure_mb': 1021.0, 'pressure_in': 30.15, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 88, 'cloud': 0, 'feelslike_c': 10.8, 'feelslike_f': 51.5, 'windchill_c': 12.3, 'windchill_f': 54.2, 'heatindex_c': 13.5, 'heatindex_f': 56.4, 'dewpoint_c': 9.2, 'dewpoint_f': 48.5, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 1.0, 'gust_mph': 14.7, 'gust_kph': 23.7}}", 'score': 0.98315, 'raw_content': None}, {'title': 'Weather in Sydney in June 2024', 'url': 'https://world-weather.info/forecast/australia/sydney/june-2024/', 'content': 'Detailed ⚡ Sydney Weather Forecast for June 2024 - day/night 🌡️ temperatures, precipitations - World-Weather.info. Add the current city. Search. Weather; Archive; Widgets °F. World; Australia; Weather in Sydney; Weather in Sydney in June 2024. ... 23 +57° +52° 24 +61° +48° 25 ...', 'score': 0.98173, 'raw_content': None}, {'title': 'Sydney weather in June 2024 | Sydney 14 day weather', 'url': 'https://www.weather25.com/oceania/australia/new-south-wales/sydney?page=month&month=June', 'content': "Our weather forecast can give you a great sense of what weather to expect in Sydney in June 2024. If you're planning to visit Sydney in the near future, we highly recommend that you review the 14 day weather forecast for Sydney before you arrive. Temperatures. 17 ° / 10 °. Rainy Days.", 'score': 0.97698, 'raw_content': None}, {'title': 'Sydney, NSW Weather Forecast June 2024: Daily Highs/Lows & Rain Trends', 'url': 'https://www.weathertab.com/en/c/e/06/commonwealth-of-australia/state-of-new-south-wales/sydney/', 'content': 'Explore comprehensive June 2024 weather forecasts for Sydney, including daily high and low temperatures, precipitation risks, and monthly temperature trends. Featuring detailed day-by-day forecasts, dynamic graphs of daily rain probabilities, and temperature trends to help you plan ahead. ... 23 61°F 46°F 16°C 8°C 09% 24 61°F 47°F 16°C 8 ...', 'score': 0.97489, 'raw_content': None}], 'response_time': 2.53}
    """

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "api_key": API_KEY,
        "query": search_query,
        "include_answer": True
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return f"An error occurred: {response.status_code}, {response.text}"

def create_model(model_name: str, model_kwargs: dict):
    print("Creating model")
    agent = reasoning_engines.LangchainAgent(
        model=model_name, 
        tools=[tavily_search_method],
        model_kwargs=model_kwargs,
    )
    return agent

def deploy_agent(agent_name: str, agent):
    remote_app = reasoning_engines.ReasoningEngine.create(
        agent,
        requirements=[
            "google-cloud-aiplatform[reasoningengine,langchain]",
        ],
        display_name=agent_name,
    )
    print(remote_app)

if __name__ == "__main__":
    init_vertexai()
    model_name = "gemini-1.0-pro"
    safety_settings = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
    model_kwargs = {
        "temperature": 0.28,
        "max_output_tokens": 1000,
        "top_p": 0.95,
        "top_k": 40,
        "safety_settings": safety_settings,
    }
    agent = create_model(model_name=model_name, model_kwargs=model_kwargs)
    response = agent.query(
        input="What is the weather in Sydney on 24th June 2024? Should I go to the office?"
    )
    print(response)
   

    if response["output"] != 'I am sorry, I cannot fulfill this request. The available tools lack the desired functionality.':
        print("Model validated")
        deploy_agent(agent_name="test_agent2", agent=agent)
    else:
        print("Will not deploy")