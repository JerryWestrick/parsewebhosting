# This is a sample Python script.
import os

os.environ["OPENAI_API_KEY"] = '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'
os.environ["SERPAPI_API_KEY"] = 'e9f8a96b4dce334dfb65cacb0d04220691fae9bd8965dbeac73593f802180039'

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

def Example_Agents():
    # First, let's load the language model we're going to use to control the agent.
    llm = OpenAI(temperature=0)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)


    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Now let's test it out!
    agent.run("At what time does the united flight from Atlanta to Cancun land this saturday?  "
              "At what timeshuld I leave Merida in a car to arrive at the cancun airport at that time?")



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def Example_Memory():
    from langchain import OpenAI, ConversationChain

    llm = OpenAI(temperature=0)
    conversation = ConversationChain(llm=llm, verbose=True)

    output = conversation.predict(input="Hi there!")
    print(output)
    output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    print(output)


def Example_Chat():
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    chat = ChatOpenAI(temperature=0)
    response = chat([HumanMessage(content="Translate this sentence from English to German. I love programming.")])
    print(f'AI: {response}')

    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to German."),
            HumanMessage(content="Translate this sentence from English to German. I love programming.")
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates English to German."),
            HumanMessage(content="Translate this sentence from English to German. I love artificial intelligence.")
        ],
    ]
    result = chat.generate(batch_messages)
    print(result)


def Example_Memory2():
    # Memory: Add State to Chains and Agents
    from langchain.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
    )
    from langchain.chains import ConversationChain
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI. "
            "The AI is talkative and provides lots of specific details from its context. "
            "If the AI does not know the answer to a question, it truthfully says it does not know."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    print(conversation.predict(input="Hi there!"))
    # -> 'Hello! How can I assist you today?'

    print(conversation.predict(input="I'm doing well! Just having a conversation with an AI."))
    # -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

    print(conversation.predict(input="Tell me about yourself."))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Ramon')
    Example_Memory2()






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
