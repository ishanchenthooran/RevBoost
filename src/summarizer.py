import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("REVBOOST_OPENAI_API_KEY"))

def generate_summary(customer_info: dict, temperature: float = 0.7) -> str:
    """
    Use OpenAI to generate a summary or recommendation for a given customer profile.

    Parameters:
        customer_info (dict): Dictionary of customer attributes
        temperature (float): Sampling temperature for creativity

    Returns:
        str: LLM-generated summary
    """
    prompt = (
        "You're a customer retention expert.\n"
        "Analyze the following customer profile and give:\n"
        "- A retention strategy\n"
        "- Any upsell opportunities\n"
        "\nCustomer Info:\n"
    )

    for key, value in customer_info.items():
        prompt += f"{key}: {value}\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=250,
    )

    # Safely handle possible None in message.content
    msg = response.choices[0].message
    text = (msg.content or "").strip()

    if not text:
        raise ValueError("⚠️ OpenAI response contained no text content.")

    return text

# Summarizer Test
# if __name__ == "__main__":
#    test_customer = {
#        "Segment": "2",
#        "Tenure": "2 months",
#        "Monthly Charges": "$80",
#        "Total Services Used": "2",
#        "Churn Probability": "0.91"
#    }
#    summary = generate_summary(test_customer)
#    print("📌 LLM Summary:\n", summary)
