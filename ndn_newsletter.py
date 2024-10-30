import praw
import time
import os
import scholarly

from langchain.tools import tool
from langchain_ollama import OllamaLLM
from crewai import Agent, Task, Process, Crew


from langchain_community.agent_toolkits.load_tools import load_tools

# To load Human in the loop
human_tools = load_tools(["human"])

# To Load GPT-4
api = os.environ.get("OPENAI_API_KEY")

# https://docs.crewai.com/concepts/llms#using-ollama-local-llms
# To Load Local models through Ollama
from crewai import LLM
llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

class BrowserTool:
    @tool("Scrape reddit content")
    def scrape_reddit(max_comments_per_post=7):
        """Useful to scrape a reddit content"""
        reddit = praw.Reddit(
            client_id="client-id",
            client_secret="client-secret",
            user_agent="user-agent",
        )
        subreddit = reddit.subreddit("LocalLLaMA")
        scraped_data = []

        for post in subreddit.hot(limit=12):
            post_data = {"title": post.title, "url": post.url, "comments": []}

            try:
                post.comments.replace_more(limit=0)  # Load top-level comments only
                comments = post.comments.list()
                if max_comments_per_post is not None:
                    comments = comments[:7]

                for comment in comments:
                    post_data["comments"].append(comment.body)

                scraped_data.append(post_data)

            except praw.exceptions.APIException as e:
                print(f"API Exception: {e}")
                time.sleep(60)  # Sleep for 1 minute before retrying

        return scraped_data


"""
- define agents that are going to research latest AI tools and write a blog about it 
- explorer will use access to internet and LocalLLama subreddit to get all the latest news
- writer will write drafts 
- critique will provide feedback and make sure that the blog text is engaging and easy to understand
"""

explorer = Agent(
    role="高级研究员",
    goal="在 LocalLLama subreddit 上发现和探索 2024 年最令人兴奋的项目和公司",
    backstory="""你是一位专业的战略分析师，擅长发现 AI、科技和机器学习领域的新兴趋势和公司。
    你特别善于在 LocalLLama subreddit 上找到有趣、令人兴奋的项目。你能够将抓取的数据转化为详细的报告，
    其中包含 AI/ML 领域最令人兴奋的项目和公司名称。仅使用从 LocalLLama subreddit 抓取的数据进行报告。
    """,
    verbose=True,
    allow_delegation=False,
    tools=[BrowserTool().scrape_reddit] + human_tools,
    llm=llm,
)

writer = Agent(
    role="高级技术写作者",
    goal="使用简单、通俗的词汇撰写关于最新 AI 项目的引人入胜的博客文章",
    backstory="""你是一位专门研究技术创新的专业写作者，特别专注于 AI 和机器学习领域。你知道如何用
    引人入胜、有趣但简单、直接和简洁的方式写作。你擅长通过使用通俗易懂的语言，以有趣的方式向普通读者
    解释复杂的技术术语。仅使用从 LocalLLama subreddit 抓取的数据撰写博客。""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

critic = Agent(
    role="专业写作评论家",
    goal="对博客文章草稿提供反馈和批评。确保文章的语气和写作风格具有吸引力、简单和简洁",
    backstory="""你是一位为技术写作者提供反馈的专家。你能够判断出博客文本是否不够简洁、
    简单或吸引人。你知道如何提供有助于改进任何文本的有用反馈。你知道如何通过使用通俗易懂的
    术语来确保文本保持技术性和深入性。
    """,
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

task_report = Task(
    description="""分析从 LocalLLama subreddit 抓取的数据，创建一份详细报告。

具体要求：
1. 仅使用从 LocalLLama 抓取的数据
2. 输出格式必须是纯文本报告
3. 使用项目符号列出 5-10 个新兴的 AI 项目和工具
4. 每个项目符号必须包含 3 个句子，专门描述一个特定的 AI 公司、产品或模型
5. 忽略任何代码或非文本内容

请直接提供最终报告，不要包含其他动作或解释。
    """,
    agent=explorer,
    expected_output="A detailed bullet-point report analyzing 5-10 emerging AI projects from LocalLLama subreddit"
)

task_blog = Task(
    description="""基于之前的报告，创建一篇博客文章。

输出格式要求：
```markdown
## [项目标题](项目链接)
- 有趣的事实
- 与整体主题的关联分析

## [第二个项目标题](项目链接)
- 有趣的事实
- 与整体主题的关联分析
```

注意事项：
1. 使用上述 markdown 格式
2. 项目名称使用粗体
3. 包含项目链接
4. 只包含来自 LocalLLama 的信息
5. 使用简单易懂的语言
    """,
    agent=writer,
    expected_output="A blog post in markdown format summarizing the AI projects findings"
)

task_critique = Task(
    description="""审查博客文章并确保其符合以下格式：
```markdown
## [项目标题](项目链接)
- 有趣的事实
- 与整体主题的关联分析

## [第二个项目标题](项目链接)
- 有趣的事实
- 与整体主题的关联分析
```

如果格式不符，请按照上述格式重写文章。只需提供最终修改后的文章，无需其他解释。
    """,
    agent=critic,
    expected_output="A reviewed and potentially revised version of the blog post maintaining the required markdown format"
)

# instantiate crew of agents
crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=True,  
    process=Process.sequential,
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)



