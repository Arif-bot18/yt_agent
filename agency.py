from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 
from crewai_tools import SerperDevTool
from crewai import Agent , Task, Crew


#take the input from user for topic
topic = input("Enter Your Topic To Search Trends: ").strip()



#load the env file api keys 
load_dotenv()

# initializing the serper (used to for search the real time web)
search_tool = SerperDevTool()

try :
    #llm model (which is brain of the agent)
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",google_api_key = os.getenv("GEMINI_API_KEY"))
    # agent 1
    # giving the persona how to act while doing the task 
    trend_analyst = Agent(role = "Senior YouTube Growth Researcher",goal = "Identify the top 3 high-performing video angles based on the current trends",
    backstory= """ You are a YouTube growth strategist who studies search trends,
        viral video patterns, and audience curiosity.
        You believe views alone do not reveal demand.
        Comment sections often reveal what viewers truly want to learn.
        Many creators ignore unanswered audience questions.
        Your job is to identify those gaps and turn them into powerful video angles.
        """,
        tools = [search_tool],
        llm = llm,
        verbose = True)

    #the task 
    trend_research_task = Task(description=f""" Research the topic {topic} and identify the top 3 YouTube video angles that could go viral. Use internet search if needed. """, 
    expected_output=""" 3 strong YouTube video ideas with a short explanation for why each idea could perform well. """,agent = trend_analyst)
    # agent 2
    # this used for creating the scripts for those ideas 
    content_architect = Agent(role = "Lead Scriptwriter & Hook Expert",
    goal= "Convert raw research into a high-retention YouTube script including a '10-second hook' and a 'call to action'.",
    backstory = "You have written scripts for channels with millions of subscribers. Your specialty is making complex technical topics (like AI) sound exciting and easy to understand.",
    llm = llm,verbose = True)
    #task
    script_reserch_task = Task(description=""" Write a complete YouTube script based on the video ideas generated in the previous task. Choose the 1 best idea and expand it into a script. """,
    expected_output="A full YouTube script with hook, body, and conclusion", agent=content_architect, output_file = "script.md")

    # crew will  manage the  workflow of this agent you have to give the agent and task in sequence order 
    crew = Crew(agents= [trend_analyst,content_architect],tasks=[trend_research_task,script_reserch_task],verbose = True)

    response = crew.kickoff()
    print(response)

except Exception as e:
    print(f"THE ERROR IS : {e}")
