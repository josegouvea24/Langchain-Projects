from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from gen_ai_hub.proxy.native.openai import chat
import datetime

#############
# 1 - SETUP #
#############

# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date

llm_model = "gpt-35-turbo"


############################
# 2 - Chat API: Gen AI Hub #
############################

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = chat.completions.create(messages=messages, model_name=llm_model, temperature=0.5)
    return response.choices[0].message.content

get_completion("What is 1+1?")

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

# response = get_completion(prompt)
# print(response)

###########################
# 2 - Chat API: LangChain #
###########################

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
my_chat = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client)
#print(my_chat)

# #######################
# # 3 - Prompt Template #
# #######################

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)

#print(prompt_template.messages[0].prompt)

customer_style = """American English \
in a calm and respectful tone
"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

#print(type(customer_messages))
#print(type(customer_messages[0]))
#print(customer_messages[0])

customer_response = my_chat(customer_messages)

#print(customer_response.content)

######################
# 4 - Output Parsers #
######################

# Example of a desired output
{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}

# Example of an input
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

# Example of a prompt template for the model output
review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
#print(prompt_template)

messages = prompt_template.format_messages(text=customer_review)
#print(my_chat(messages).content)

########################################################
# 4 - Parse LLM output string into a Python dictionary #
########################################################

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# Prompt template with format instructions
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(review_template_2)
messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)

print(messages[0].content)

response = my_chat(messages)
print(response.content)

output_dic = output_parser.parse(response.content)
print(output_dic)
# Output dictionary is now type dict
print( type(output_dic) )

# We can now access the values of the dictionary individually
print(output_dic.get('delivery_days'))