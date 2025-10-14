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
    "You're a customer success strategist at a SaaS company analyzing customer data for churn prevention and revenue expansion.\n"
    "Use the profile below to write a short, structured strategy with 2 parts:\n\n"
    "1) **Retention Strategy** — specific business actions to reduce churn risk, referencing contract type, tenure, and engagement level.\n"
    "   - Mention which data signals indicate loyalty or risk.\n"
    "   - Recommend tactical steps (e.g., personalized outreach, loyalty credits, usage-based incentives, renewal timing, customer education).\n\n"
    "2) **Upsell Opportunities** — realistic cross-sell or upsell ideas based on spending and service usage.\n"
    "   - Mention financial rationale and customer behavior logic.\n"
    "   - Suggest timing or offer types (e.g., premium bundles, feature add-ons, referral programs).\n\n"
    "Keep the tone professional yet consultative. Focus on business impact and customer lifetime value.\n"
    "Limit total output to 6–8 sentences.\n\n"
    f"Customer Profile:\n{customer_info}\n"
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


