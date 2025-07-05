from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.panel import Panel

from constant import DOCKER_WORKPLACE_NAME, COMPLETION_MODEL
from datasets import load_dataset
from autoagent import MetaChain
from autoagent.agents import get_system_triage_agent
from autoagent.logger import MetaChainLogger, LoggerManager
from evaluation.utils import check_port_available
import os.path as osp
import re
import os
from autoagent.environment.docker_env import DockerEnv, DockerConfig, check_container_ports
from autoagent.environment.markdown_browser import RequestsMarkdownBrowser
from autoagent.types import Response
from autoagent.cli_utils.file_select import select_and_copy_files
from autoagent.util import single_select_menu
from autoagent.environment.local_env import LocalEnv
from autoagent.environment.browser_env import BrowserEnv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

def clear_screen():
    console = Console()
    console.print("[bold green]Coming soon...[/bold green]")
    print('\033[u\033[J\033[?25h', end='')  # Restore cursor and clear everything after it, show cursor
def get_config(container_name, port, test_pull_name="main", git_clone=False):
    container_name = container_name
    
    port_info = check_container_ports(container_name)
    if port_info:
        port = port_info[0]
    else:
        # while not check_port_available(port):
        #     port += 1
        # 使用文件锁来确保端口分配的原子性
        import filelock
        lock_file = os.path.join(os.getcwd(), ".port_lock")
        lock = filelock.FileLock(lock_file)
        
        with lock:
            port = port
            while not check_port_available(port):
                port += 1
                print(f'{port} is not available, trying {port+1}')
            # 立即标记该端口为已使用
            with open(os.path.join(os.getcwd(), f".port_{port}"), 'w') as f:
                f.write(container_name)
    local_root = os.path.join(os.getcwd(), f"workspace_meta_showcase", f"showcase_{container_name}")
    os.makedirs(local_root, exist_ok=True)
    docker_config = DockerConfig(
        workplace_name=DOCKER_WORKPLACE_NAME,
        container_name=container_name,
        communication_port=port,
        conda_path='/root/miniconda3',
        local_root=local_root,
        test_pull_name=test_pull_name,
        git_clone=git_clone
    )
    return docker_config
def create_environment(docker_config: DockerConfig):
    """
    1. create the code environment
    2. create the web environment
    3. create the file environment
    """
    code_env = DockerEnv(docker_config)
    code_env.init_container()
    
    web_env = BrowserEnv(browsergym_eval_env = None, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name)
    file_env = RequestsMarkdownBrowser(viewport_size=1024 * 5, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name, downloads_folder=os.path.join(docker_config.local_root, docker_config.workplace_name, "downloads"))
    
    return code_env, web_env, file_env

def create_environment_local(docker_config: DockerConfig):
    """
    1. create the code environment
    2. create the web environment
    3. create the file environment
    """
    code_env = LocalEnv(docker_config)

    web_env = BrowserEnv(browsergym_eval_env = None, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name)
    file_env = RequestsMarkdownBrowser(viewport_size=1024 * 5, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name, downloads_folder=os.path.join(docker_config.local_root, docker_config.workplace_name, "downloads"))
    
    return code_env, web_env, file_env


def main(container_name: str, port: int, test_pull_name: str, git_clone: bool, local_env: bool):
    """
    Run deep research with a given model, container name, port
    """ 
    model = COMPLETION_MODEL
    print('\033[s\033[?25l', end='')  # Save cursor position and hide cursor
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True  # 这会让进度条完成后消失
    ) as progress:
        task = progress.add_task("[cyan]Initializing...", total=None)
        
        progress.update(task, description="[cyan]Initializing config...[/cyan]\n")
        docker_config = get_config(container_name, port, test_pull_name, git_clone)
        
        progress.update(task, description="[cyan]Setting up logger...[/cyan]\n")
        log_path = osp.join("casestudy_results", 'logs', f'agent_{container_name}_{model}.log')
        LoggerManager.set_logger(MetaChainLogger(log_path = None))
        
        progress.update(task, description="[cyan]Creating environment...[/cyan]\n")
        if local_env:
            code_env, web_env, file_env = create_environment_local(docker_config)
        else:
            code_env, web_env, file_env = create_environment(docker_config)
        
        progress.update(task, description="[cyan]Setting up autoagent...[/cyan]\n")
    
    clear_screen()

    context_variables = {"working_dir": docker_config.workplace_name, "code_env": code_env, "web_env": web_env, "file_env": file_env}

    # select the mode
    while True:
        # update_guidance(context_variables)
        mode = single_select_menu([ 'evaluation mode', 'exit'], "Please select the mode:")
        match mode:
            case 'evaluation mode':
                clear_screen()
                user_mode_with_evaluation(model, context_variables, False)
            case 'exit':
                console = Console()
                logo_text = Text('Good bye', justify="center")
                console.print(Panel(logo_text, style="bold salmon1", expand=True))
                break

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=10, max=180),
    before_sleep=lambda retry_state: print(f"\033[91mRetrying... (attempt {retry_state.attempt_number})\033[0m"),
)
def user_mode(model: str, context_variables: dict, debug: bool = True, query: str = 'exit', count: int = 0) -> bool:
    logger = LoggerManager.get_logger()
    console = Console()
    system_triage_agent = get_system_triage_agent(model) # System triage agent is responsible for delegating tasks
    assert system_triage_agent.agent_teams != {}, "System Triage Agent must have agent teams"
    messages = []
    agent = system_triage_agent
    # This is for debugging
    # print(f"\033[91m[Location 0]:\033[0m agent = {agent}")
    agents = {system_triage_agent.name.replace(' ', '_'): system_triage_agent}
    for agent_name in system_triage_agent.agent_teams.keys():
        agents[agent_name.replace(' ', '_')] = system_triage_agent.agent_teams[agent_name]("placeholder").agent
    agents["Upload_files"] = "select"
    
    client = MetaChain(log_path=logger)
    upload_infos = []
    
    if query.strip().lower() == 'exit':
        # logger.info('User mode completed. See you next time! :waving_hand:', color='green', title='EXIT')
        logo_text = "User mode completed. See you next time! :waving_hand:"
        console.print(Panel(logo_text, style="bold salmon1", expand=True))
    
    words = query.split()
    console.print(f"[bold green]Your request: {query}[/bold green]", end=" ")
    # Extract agent mentioned by user
    for word in words:
        if word.startswith('@') and word[1:] in agents.keys():
            # print(f"[bold magenta]{word}[bold magenta]", end=' ') 
            agent = agents[word.replace('@', '')]
            print(f"\033[91m[Location 1_in_if]:\033[0m agent = {agent}")
        else:
            # print(word, end=' ')
            pass
    print(f"\033[91m[Location Final]:\033[0m agent = {agent}")
    print()
    
    case_resolved = False
    if hasattr(agent, "name"): # The hasattr() function returns True if the specified object has the specified attribute, otherwise False.
        agent_name = agent.name
        console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] will help you, be patient...[/bold green]")
        if len(upload_infos) > 0:
            query = "{}\n\nUser uploaded files:\n{}".format(query, "\n".join(upload_infos))
        messages.append({"role": "user", "content": query})
        # Original code:
        response = client.run(agent, messages, context_variables, debug=debug)
        messages.extend(response.messages) # Response(messages=history[init_len:], ...)
        
        model_answer_raw = messages[-1]['content'] # Get the last response in the histroy? Yes, this is the last response
        if model_answer_raw.lstrip(' ').startswith('Case resolved'):
            model_answer = re.findall(r'<solution>(.*?)</solution>', model_answer_raw, re.DOTALL)
            if len(model_answer) == 0:
                model_answer = model_answer_raw
            else:
                model_answer = model_answer[0]
            case_resolved = True
        else: 
            model_answer = re.findall(r'<solution>(.*?)</solution>', model_answer_raw, re.DOTALL)
            if len(model_answer) == 0:
                model_answer = model_answer_raw
            else:
                model_answer = model_answer[0]
                case_resolved = True
        
        # Write the model_answer to files for analysis ------------------------------------------------------------------------------------------
        txt_file = os.path.join('/home/olina/Documents/AutoAgentEnv/Testing', f"{count}.txt")
        try:
            with open(txt_file, "w") as f:
                f.write(model_answer)
        except FileNotFoundError:
            os.system(f"mkdir -p {'/home/olina/Documents/AutoAgentEnv/Testing'}")
            with open(txt_file, "w") as f:
                f.write(model_answer)

        console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] has finished with the response:\n[/bold green] [bold blue]{model_answer}[/bold blue]")
        # agent = response.agent # Update activate agent, comment out because our evaluatino does not need to deal with multiple queries
    elif agent == "select": 
        # Why the model will enter this branch? It should not be accessed!
        print("\033[32m[INFO] Found agent==\"select\"\033[0m")
        code_env: DockerEnv = context_variables["code_env"]
        local_workplace = code_env.local_workplace
        docker_workplace = code_env.docker_workplace
        files_dir = os.path.join(local_workplace, "files")
        docker_files_dir = os.path.join(docker_workplace, "files")
        os.makedirs(files_dir, exist_ok=True)
        upload_infos.extend(select_and_copy_files(files_dir, console, docker_files_dir))
        # agent = agents["System_Triage_Agent"] # Change to code environment, and update agent to system triage agent
    else: 
        console.print(f"[bold red]Unknown agent: {agent}[/bold red]")
    
    return case_resolved

def user_mode_with_evaluation(model: str, context_variables: dict, debug: bool = True):
    # Count simpleQA problems
    ds = load_dataset("basicv8vc/SimpleQA")
    split_simpleQ = ds['test']["problem"]
    print(f"Total count of simpleQA questions: {len(split_simpleQ)}")
    count = int(input("Start from: "))
    endcount = int(input("End at(included): "))
    for i in range(count, endcount+1):
        simpleQ = split_simpleQ[i]
        print("**"*15 + f"Question {i} in range({count}, {endcount})" + '**'*15)
        for attempt in range(3):
            print(f"\033[32m[INFO] Attempt: {attempt+1}")
            resolved = user_mode(model, context_variables, debug, simpleQ, i) # Try to deal with error in browsing web. Those errors will lead to strange reply of LLM models
            if resolved:
                break
        print("**"*70)
    user_mode(model, context_variables, debug, 'exit')

    

if __name__ == '__main__':
    # container_name: str, port: int, test_pull_name: str, git_clone: bool, local_env: bool
    main(
        container_name='auto_agent_eval',
        port=12347,
        test_pull_name='autoagent_mirror',
        git_clone=True,
        local_env=False
    )