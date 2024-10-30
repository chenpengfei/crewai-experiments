import scholarly
import time
import os
from crewai import Agent, Task, Process, Crew
from langchain.tools import tool
from crewai import LLM
from langchain_community.agent_toolkits.load_tools import load_tools

# 配置 LLM
llm = LLM(model="ollama/qwen2.5:7b", base_url="http://localhost:11434")
human_tools = load_tools(["human"])

class ScholarTool:
    @tool("搜索 Google Scholar")
    def search_scholar(self, max_results=20):
        """搜索 NDN 相关的学术文献"""
        search_queries = [
            '"Named Data Networking" AND ("high performance forwarding" OR DPDK)',
            'NDN AND ("high performance forwarding" OR DPDK)',
            '"Named Data Networking" performance optimization'
        ]
        
        papers = []
        for query in search_queries:
            search_query = scholarly.search_pubs(query)
            try:
                for i in range(max_results // len(search_queries)):
                    paper = next(search_query)
                    paper_data = {
                        "title": paper.bib.get('title'),
                        "authors": paper.bib.get('author', []),
                        "abstract": paper.bib.get('abstract', ''),
                        "year": paper.bib.get('year'),
                        "url": paper.bib.get('url', ''),
                        "doi": paper.bib.get('doi', ''),
                        "citations": paper.citedby,
                        "venue": paper.bib.get('venue', ''),
                        "scholar_url": f"https://scholar.google.com/scholar?cluster={paper.cluster_id}" if hasattr(paper, 'cluster_id') else '',
                        "pdf_url": paper.bib.get('eprint', '')
                    }
                    papers.append(paper_data)
                    time.sleep(2)
            except StopIteration:
                continue
            except Exception as e:
                print(f"搜索错误: {e}")
                time.sleep(10)
                
        return papers

# 定义智能体
researcher = Agent(
    role="NDN研究专家",
    goal="分析 NDN 领域的最新研究进展，特别关注高性能转发和 DPDK 相关技术",
    backstory="""你是一位资深的 NDN 研究专家，对网络架构和高性能转发技术有深入的理解。
    你擅长分析学术论文，能够识别重要的研究趋势和技术创新。你特别关注 NDN 中的高性能转发实现，
    以及如何使用 DPDK 等技术来提升 NDN 的性能。""",
    verbose=True,
    allow_delegation=False,
    tools=[ScholarTool().search_scholar] + human_tools,
    llm=llm
)

analyzer = Agent(
    role="技术分析师",
    goal="对研究成果进行深入分析，提炼关键技术要点",
    backstory="""你是一位专业的技术分析师，擅长解读复杂的技术文献，并提取其中的核心价值。
    你能够理解不同技术方案的优劣，并善于进行横向对比。你对高性能网络技术有深入了解，
    能够准确评估不同实现方案的技术特点。""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

writer = Agent(
    role="技术文档撰写专家",
    goal="将研究发现整理成清晰的技术报告",
    backstory="""你是一位经验丰富的技术文档作者，擅长将复杂的研究内容转化为清晰的技术报告。
    你知道如何组织内容层次，突出重点，并用准确的技术语言描述研究成果。你特别善于总结技术趋势
    和未来发展方向。""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# 定义任务
task_research = Task(
    description="""分析从 Google Scholar 获取的 NDN 研究文献，创建研究报告。

要求：
1. 重点关注最近 3 年的研究成果
2. 特别关注高性能转发和 DPDK 相关研究
3. 分析内容包括：
   - 研究热点分布
   - 关键技术方案
   - 性能优化方法
4. 使用中文输出
5. 必须包含原文链接

输出格式：
```markdown
## 研究概况
[总体分析]

## 技术分类
### 高性能转发技术
#### [论文标题](论文链接)
- 发表年份：YYYY
- 主要贡献：
- 技术方案：
- 实验结果：

### DPDK 应用研究
#### [论文标题](论文链接)
- 发表年份：YYYY
- 主要贡献：
- 技术方案：
- 实验结果：

### 其他优化方案
[类似格式]
```
    """,
    agent=researcher,
    expected_output="一份包含 NDN 研究文献分析的详细报告，重点关注高性能转发和 DPDK 相关研究"
)

task_analysis = Task(
    description="""对研究成果进行深入分析。

要求：
1. 对比不同技术方案的优劣
2. 分析实现难度和部署成本
3. 评估性能提升效果
4. 使用中文输出
5. 引用具体论文时必须包含链接

输出格式：
```markdown
## 技术方案对比
### 方案类别一
- 代表性工作：[论文标题](论文链接)
- 优势：
- 劣势：
- 适用场景：

### 方案类别二
[类似格式]

## 性能评估
### [评估维度]
- 最佳方案：[论文标题](论文链接)
- 性能数据：
- 评估结论：

## 实现难度分析
[分析内容，引用具体论文时带链接]

## 部署建议
[建议内容，引用具体论文时带链接]
```
    """,
    agent=analyzer,
    expected_output="一份对 NDN 研究成果的深入技术分析报告，包含技术方案对比和性能评估"
)

task_report = Task(
    description="""生成最终技术报告。

要求：
1. 整合研究发现和分析结果
2. 突出技术创新点
3. 预测发展趋势
4. 使用中文输出
5. 所有引用必须包含原文链接

输出格式：
```markdown
# NDN 高性能转发技术研究报告

## 研究现状
### 最新进展
- [论文标题](论文链接)：主要发现
- [论文标题](论文链接)：主要发现

## 关键技术
### [技术类别]
#### 技术原理
[描述]

#### 代表性工作
1. [论文标题](论文链接)
   - 创新点：
   - 实现方式：
   - 性能指标：

## 性能优化方案
### DPDK 相关优化
1. [论文标题](论文链接)
   - 优化方法：
   - 性能提升：
   - 部署难度：

### 其他优化方案
[类似格式]

## 发展趋势
[趋势分析，引用支撑论文]

## 研究建议
1. [建议主题]
   - 依据：[相关研究](论文链接)
   - 具体建议：
```
    """,
    agent=writer,
    expected_output="一份完整的 NDN 高性能转发技术研究报告，包含研究现状、关键技术、性能优化方案和发展趋势"
)

# 创建和启动 Crew
crew = Crew(
    agents=[researcher, analyzer, writer],
    tasks=[task_research, task_analysis, task_report],
    verbose=True,
    process=Process.sequential
)

# 执行任务
result = crew.kickoff()

print("\n=================== 最终报告 ===================\n")
print(result) 