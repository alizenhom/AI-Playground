import json
import os
from typing import List
from crewai import Agent, Process, Task, Crew, LLM
from crewai.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tavily import TavilyClient
from scrapegraph_py import Client


load_dotenv()

output_dir = "./ai-agent-output"
basic_llm = LLM(model="gpt-4o", temperature=0)
groq_llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0)
llm = groq_llm
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
scrapegraph_client = Client(os.getenv("SCRAPEGRAPH_API_KEY"))


# Agent A
class SearchQueries(BaseModel):
    search_queries: List[str] = Field(
        ...,
        description="A list of search queries for the product.",
        min_items=1,
        max_items=10,
    )


search_queries_recommender_agent = Agent(
    role="Search Queries Recommender Agent",
    goal="\n".join(
        [
            "To generate search queries for a given product to be passed to a search engine.",
            "The search queries should be relevant to the product and should be able to find",
        ]
    ),
    backstory="The agent is designed to generate a list of search queries for a given product to be passed to a search engine.",
    llm=llm,
    verbose=True,
)


search_queries_recommender_task = Task(
    description="\n".join(
        [
            "My company is looking to purchase a {product_name} at the best prices (value for money).",
            "You're allowed to search products from curated list of websites given here: {websites_list}",
            "The search queries should be relevant to the product and should be able to find to be compared later in another agent.",
            "All stores are based in {delivery_country}.",
            "Generate {number_of_queries} search queries for the product.",
            "The search query must reach a product e-commerce webpage for the product not just a blog post or a news article.",
            "The search queries must include the website domain name.",
            "The search queries should be in {language}.",
        ]
    ),
    expected_output="A json object containing the list of search queries.",
    output_json=SearchQueries,
    output_file=f"{os.path.join(output_dir, 'search_queries.json')}",
    agent=search_queries_recommender_agent,
)


# Agent B
class SearchResult(BaseModel):
    title: str = Field(..., description="The title of the search result.")
    url: str = Field(..., description="The url of the search result.")
    content: str = Field(..., description="The content of the search result.")
    confidence_score: float = Field(
        ..., description="The confidence score of the search result."
    )
    search_query: str = Field(
        ..., description="The search query that was used to find the result."
    )


class SearchResults(BaseModel):
    search_results: List[SearchResult] = Field(
        ...,
        description="A list of search results.",
        min_items=1,
        max_items=10,
    )


@tool
def search_engine_tool(query: str) -> str:
    """
    Search the web for the given query.
    """
    return tavily_client.search(query=query)


search_engine_agent = Agent(
    role="Search Engine Agent",
    goal="\n".join(
        [
            "To search the web for the given query.",
        ]
    ),
    backstory="The agent is designed to search the web for the given query.",
    llm=llm,
    verbose=True,
    tools=[search_engine_tool],
)


search_engine_task = Task(
    description="\n".join(
        [
            "The task is to search for products based on the search queries provided by the Search Queries Recommender Agent.",
            "You have to collect results from multiple search queries.",
            "Ignore any results with confidence score less than {confidence_score}.",
            "Ignore any results that are not direct links to a single product page.",
            "The search results will be used to compare prices of the product across different websites.",
        ]
    ),
    expected_output="A json object containing the search results.",
    output_json=SearchResults,
    output_file=f"{os.path.join(output_dir, 'search_results.json')}",
    agent=search_engine_agent,
)


class ProductSpec(BaseModel):
    name: str = Field(..., description="The name of the product specification.")
    value: str = Field(..., description="The value of the product specification.")


class HtmlScraperResult(BaseModel):
    page_url: str = Field(..., description="The url of the product page.")
    product_title: str = Field(..., description="The title of the product.")
    product_price: str = Field(..., description="The price of the product.")
    old_product_price: str = Field(
        description="The old price of the product.", default=None
    )
    product_image_url: str = Field(
        description="The image url of the product.", default=None
    )
    product_discount: str = Field(
        description="The discount of the product.", default=None
    )
    product_specs: List[ProductSpec] = Field(
        ...,
        description="The specifications of the product. Focus on the most important features.",
        min_items=1,
        max_items=5,
    )

    agent_recommendation_rank: int = Field(
        ...,
        description="The rank of the product based on the agent's recommendation. This will be used to rank the products. The higher the rank, the more recommended the product is. (0 is lowest, 5 is highest)",
    )
    agent_recommendation_reason: str = Field(
        ...,
        description="The reason for the agent's recommendation. This will be used to justify the rank of the product.",
    )


class HtmlScraperResults(BaseModel):
    products: List[HtmlScraperResult] = Field(
        ...,
        description="A list of html scraper results.",
        min_items=1,
        max_items=10,
    )


# Agent C
@tool
def html_scraper_tool(url: str) -> str:
    """
    Scrape the html of the given url.
    Example:
    html_scraper_tool(
        url = "https://www.amazon.eg/-/en/Mienta-american-coffee-barista-cm31316a/dp/B082VXBZYX",
    )
    """
    details = scrapegraph_client.smartscraper(
        website_url=url,
        user_prompt=f"Extract the following fields: ```json\n{HtmlScraperResult.model_json_schema()}\n``` from the html of the given url.",
    )
    return {
        "url": url,
        "details": details,
    }


html_scraper_agent = Agent(
    role="HTML Scraper Agent",
    goal="\n".join(
        [
            "The task is to scrape the html of the given url.",
        ]
    ),
    backstory="The agent is designed to scrape the html of the given url and extract the product details. These details will be used to compare prices of the product across different websites.",
    llm=llm,
    verbose=True,
    tools=[html_scraper_tool],
)
html_scraper_task = Task(
    description="\n".join(
        [
            "The task is to scrape the html of the given url.",
            "The task has to collect results from multiple urls.",
        ]
    ),
    expected_output="A json object containing the html of the given url.",
    output_json=HtmlScraperResults,
    output_file=f"{os.path.join(output_dir, 'html_scraper_results.json')}",
    agent=html_scraper_agent,
)


# Agent D


report_generator_agent = Agent(
    role="Report Generator Agent",
    goal="\n".join(
        [
            "to generate a markdown report from the given html scraper results.",
        ]
    ),
    backstory="The agent is designed to generate a markdown report from the given html scraper results.",
    llm=llm,
    verbose=True,
)


report_generator_task = Task(
    description="\n".join(
        [
            "The task is to generate a markdown report from the given html scraper results.",
            "The report should be structured in a way that is easy to understand and use.",
            "The report should be structured in the following sections:",
            "1. Executive Summary: A summary of the product and the market research.",
            "2. Introduction: A brief introduction to the product and the market research.",
            "3. Product Specifications: A detailed description of the product’s features, models, and technical details.",
            "4. Pros and Cons: A bullet-point list of advantages and disadvantages.",
            "5. Use Cases & Applications: Typical use cases, user demographics, and industry applications.",
            "6. Pricing & Availability: Pricing models, availability, and distribution channels.",
            "7. Conclusion & Recommendations: Final thoughts, key takeaways, and actionable insights.",
            "8. References: Sources and links used for gathering the research data.",
        ]
    ),
    expected_output="A markdown report from the given html scraper results.",
    output_file=f"{os.path.join(output_dir, 'report.md')}",
    agent=report_generator_agent,
)


def main():
    crew = Crew(
        agents=[
            search_queries_recommender_agent,
            search_engine_agent,
            html_scraper_agent,
            report_generator_agent,
        ],
        tasks=[
            search_queries_recommender_task,
            search_engine_task,
            html_scraper_task,
            report_generator_task,
        ],
        process=Process.sequential,
    )
    inputs = {
        "product_name": "coffee machine",
        "websites_list": [
            "https://www.amazon.eg",
            "https://www.jumia.com.eg",
            "https://noon.com/egypt-en",
        ],
        "delivery_country": "Egypt",
        "number_of_queries": 10,
        "language": "English",
        "confidence_score": 70,
    }
    crew.kickoff(inputs=inputs)


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
