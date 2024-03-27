import gradio as gr
import random
import wikipedia
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
#from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.schema.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

# add useful tools
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

# convert tools to functions
tools = [search_wikipedia]
functions = [convert_to_openai_function(f) for f in tools]

# build agent chain
model = ChatOpenAI(temperature=0).bind(functions=functions)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
# memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

# define agent executor
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

# define the respond function
def respond(message, chat_history):
    #No LLM here, just respond with a random pre-made message
    chat_history_message = ChatMessageHistory()
    for item in chat_history:
        chat_history_message.add_user_message(item[0])
        chat_history_message.add_ai_message(item[1])
    bot_message = agent_executor.invoke({"input": message, "chat_history":chat_history_message.messages})["output"]
    chat_history.append((message, bot_message))
    return "", chat_history

# define run agent
# def run_agent(user_input):
#     intermediate_steps = []
#     while True:
#         bot_message = agent_chain.invoke({
#             "input": user_input, 
#             "intermediate_steps": intermediate_steps
#         })
#         if isinstance(result, AgentFinish):
#             return result
#         tool = {
#             "search_wikipedia": search_wikipedia, 
#         }[result.tool]
#         observation = tool.run(result.tool_input)
#         intermediate_steps.append((result, observation))

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

demo.launch()
