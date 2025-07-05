from autoagent.types import Agent
from autoagent.registry import register_agent
from autoagent.tools import click, page_down, page_up, history_back, history_forward, web_search, input_text, sleep, visit_url, get_page_markdown
from autoagent.tools.web_tools import with_env
from autoagent.environment.browser_env import BrowserEnv
import time
from constant import DOCKER_WORKPLACE_NAME, LOCAL_ROOT
@register_agent(name = "Web Surfer Agent", func_name="get_websurfer_agent")
def get_websurfer_agent(model: str = "gpt-4o", **kwargs):
    
    def handle_mm_func(tool_name, tool_args):
        return f"After take last action `{tool_name}({tool_args})`, the image of current page is shown below. Please take next action based on the image, the current state of the page as well as previous actions and observations."
    def instructions(context_variables):
        web_env: BrowserEnv = context_variables.get("web_env", None)
        assert web_env is not None, "web_env is required"
        return f"""Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

Note that if you want to analyze the YouTube video, Wikipedia page, or other pages that contain media content, or you just want to analyze the text content of the page in a more detailed way, you should use `get_page_markdown` tool to convert the page information to markdown text. And when browsing the web, if you have downloaded any files, the path of the downloaded files will be `{web_env.docker_workplace}/downloads`, and you CANNOT open the downloaded files directly, you should transfer back to the `System Triage Agent`, and let `System Triage Agent` to transfer to `File Surfer Agent` to open the downloaded files.

Never use transfer_back_to_triage_agent until the task is fully complete. 

When you think you have completed the task the `System Triage Agent` asked you to do, you should use `transfer_back_to_triage_agent` to transfer the conversation back to the `System Triage Agent`. And you should not stop to try to solve the user's request by transferring to `System Triage Agent` only until the task is completed.

You may execute `history_back` function if an error occurs in previous tool calls.

You MUST provide the best possible next action among `click`, `page_down`, `page_up`, `history_back`, `history_forward`, `web_search`, `input_text`, `sleep`, `visit_url`, and `get_page_markdown` for each query.
"""
    
    tool_list = [click, page_down, page_up, history_back, history_forward, web_search, input_text, sleep, visit_url, get_page_markdown]
    return Agent(
        name="Web Surfer Agent", 
        model=model, 
        instructions=instructions,
        functions=tool_list,
        handle_mm_func=handle_mm_func,
        tool_choice = "required", 
        parallel_tool_calls = False
    )

"""
Note that when you need to download something, you should first know the url of the file, and then use the `visit_url` tool to download the file. For example, if you want to download paper from 'https://arxiv.org/abs/2310.13023', you should use `visit_url('url'='https://arxiv.org/pdf/2310.13023.pdf')`.
"""
# Below are original instructions

"""Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

Note that if you want to analyze the YouTube video, Wikipedia page, or other pages that contain media content, or you just want to analyze the text content of the page in a more detailed way, you should use `get_page_markdown` tool to convert the page information to markdown text. And when browsing the web, if you have downloaded any files, the path of the downloaded files will be `{web_env.docker_workplace}/downloads`, and you CANNOT open the downloaded files directly, you should transfer back to the `System Triage Agent`, and let `System Triage Agent` to transfer to `File Surfer Agent` to open the downloaded files.

When you think you have completed the task the `System Triage Agent` asked you to do, you should use `transfer_back_to_triage_agent` to transfer the conversation back to the `System Triage Agent`. And you should not stop to try to solve the user's request by transferring to `System Triage Agent` only until the task is completed.
"""