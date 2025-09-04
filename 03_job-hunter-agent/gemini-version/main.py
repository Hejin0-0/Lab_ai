import dotenv
import os

dotenv.load_dotenv()

# Configure Google Gemini
from crewai.llm import LLM

# Initialize Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, task, agent, crew
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from models import JobList, RankedJobList, ChosenJob
from tools import web_search_tool

# Try to use simple text reading instead of knowledge source to avoid OpenAI embedding issues
import pathlib

def read_resume():
    """Read resume content as simple text"""
    # Try multiple possible paths
    possible_paths = [
        pathlib.Path("knowledge/resume.txt"),
        pathlib.Path("resume.txt"),
    ]
    
    for resume_path in possible_paths:
        if resume_path.exists():
            print(f"Found resume at: {resume_path}")
            return resume_path.read_text(encoding='utf-8')
    
    print("Resume file not found in any of the expected locations")
    return ""

resume_content = read_resume()
print(f"Resume loaded successfully: {len(resume_content)} characters")

@CrewBase
class JobHunterCrew:

    @agent
    def job_search_agent(self):
        return Agent(
            config=self.agents_config["job_search_agent"],
            tools=[web_search_tool],
            llm=gemini_llm,
        )

    @agent
    def job_matching_agent(self):
        # Add resume content directly to the agent's backstory/context
        agent_config = self.agents_config["job_matching_agent"].copy()
        agent_config["backstory"] += f"\n\nUser's Resume:\n{resume_content}"
        
        return Agent(
            config=agent_config,
            llm=gemini_llm,
        )

    @agent
    def resume_optimization_agent(self):
        # Add resume content directly to the agent's backstory/context
        agent_config = self.agents_config["resume_optimization_agent"].copy()
        agent_config["backstory"] += f"\n\nUser's Original Resume:\n{resume_content}"
        
        return Agent(
            config=agent_config,
            llm=gemini_llm,
        )

    @agent
    def company_research_agent(self):
        # Add resume content directly to the agent's backstory/context
        agent_config = self.agents_config["company_research_agent"].copy()
        agent_config["backstory"] += f"\n\nUser's Resume for Context:\n{resume_content}"
        
        return Agent(
            config=agent_config,
            tools=[web_search_tool],
            llm=gemini_llm,
        )

    @agent
    def interview_prep_agent(self):
        # Add resume content directly to the agent's backstory/context
        agent_config = self.agents_config["interview_prep_agent"].copy()
        agent_config["backstory"] += f"\n\nUser's Resume:\n{resume_content}"
        
        return Agent(
            config=agent_config,
            llm=gemini_llm,
        )

    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config["job_extraction_task"],
            output_pydantic=JobList,
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config["job_matching_task"],
            output_pydantic=RankedJobList,
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config["job_selection_task"],
            output_pydantic=ChosenJob,
        )

    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config["resume_rewriting_task"],
            context=[
                self.job_selection_task(),
            ],
        )

    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config["company_research_task"],
            context=[
                self.job_selection_task(),
            ],
        )

    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config["interview_prep_task"],
            context=[
                self.job_selection_task(),
                self.resume_rewriting_task(),
                self.company_research_task(),
            ],
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Job Hunter Crew...")
    print("="*60 + "\n")
    
    result = (
        JobHunterCrew()
        .crew()
        .kickoff(
            inputs={
                "level": "Beginner",
                "position": "Full Stack Developer",
                "location": "Japan",
            }
        )
    )
