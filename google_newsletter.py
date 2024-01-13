import os

from langchain.agents import Tool
from langchain.agents import load_tools

from crewai import Agent, Task, Process, Crew
from langchain.utilities import GoogleSerperAPIWrapper

# to get your api key for free, visit and signup: https://serper.dev/
os.environ["SERPER_API_KEY"] = "serp-api-here"

search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="useful for when you need to ask the agent to search the internet",
)

# Loading Human Tools
human_tools = load_tools(["human"])

# To Load GPT-4
api = os.environ.get("OPENAI_API_KEY")


"""
- define agents that are going to research latest AI tools and write a blog about it 
- explorer will use access to internet to get all the latest news
- writer will write drafts 
- critique will provide feedback and make sure that the blog text is engaging and easy to understand
"""
explorer = Agent(
    role="Senior Researcher",
    goal="Find and explore the most exciting projects and companies in the ai and machine learning space in 2024",
    backstory="""You are and Expert strategist that knows how to spot emerging trends and companies in AI, tech and machine learning. 
    You're great at finding interesting, exciting projects on LocalLLama subreddit. You turned scraped data into detailed reports with names
    of most exciting projects an companies in the ai/ml world. ONLY use scraped data from the internet for the report.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
)

writer = Agent(
    role="Senior Technical Writer",
    goal="Write engaging and interesting blog post about latest AI projects using simple, layman vocabulary",
    backstory="""You are an Expert Writer on technical innovation, especially in the field of AI and machine learning. You know how to write in 
    engaging, interesting but simple, straightforward and concise. You know how to present complicated technical terms to general audience in a 
    fun way by using layman words.ONLY use scraped data from the internet for the blog.""",
    verbose=True,
    allow_delegation=True,
)
critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple and concise",
    backstory="""You are an Expert at providing feedback to the technical writers. You can tell when a blog text isn't concise,
    simple or engaging enough. You know how to provide helpful feedback that can improve any text. You know how to make sure that text 
    stays technical and insightful by using layman terms.
    """,
    verbose=True,
    allow_delegation=True,
)

task_report = Task(
    description="""Use and summarize scraped data from the internet to make a detailed report on the latest rising projects in AI. Use ONLY 
    scraped data to generate the report. Your final answer MUST be a full analysis report, text only, ignore any code or anything that 
    isn't text. The report has to have bullet points and with 5-10 exciting new AI projects and tools. Write names of every tool and project. 
    Each bullet point MUST contain 3 sentences that refer to one specific ai company, product, model or anything you found on the internet.  
    """,
    agent=explorer,
)

task_blog = Task(
    description="""Write a blog article with text only and with a short but impactful headline and at least 10 paragraphs. Blog should summarize 
    the report on latest ai tools found on the internet. Style and tone should be compelling and concise, fun, technical but also use 
    layman words for the general public. Name specific new, exciting projects, apps and companies in AI world. Don't 
    write "**Paragraph [number of the paragraph]:**", instead start the new paragraph in a new line. Write names of projects and tools in BOLD.
    ALWAYS include links to projects/tools/research papers.
    """,
    agent=writer,
)

task_critique = Task(
    description="""Identify parts of the blog that aren't written concise enough and rewrite and change them. Make sure that the blog has engaging 
    headline with 30 characters max, and that there are at least 10 paragraphs. Blog needs to be rewritten in such a way that it contains specific 
    names of models/companies/projects but also explanation of WHY a reader should be interested to research more. Always include links to each paper/
    project/company
    """,
    agent=critic,
)

# instantiate crew of agents
crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
