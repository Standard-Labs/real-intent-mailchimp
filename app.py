import streamlit as st
import pandas as pd
from typing import Any, Tuple

from mailchimp_marketing import Client
from mailchimp_marketing.api_client import ApiClientError
import requests
import base64
import json

from lead_tagger import BaseTagger, StandardTagger, Custom_1_Tagger

from streamlit_sortables import sort_items



# -- Constants --
default_columns = [
    'first_name',
    'last_name',
    'email_1',
    'email_2',
    'email_3',
    'phone_1',
    'phone_1_dnc',
    'phone_2',
    'phone_2_dnc',
    'phone_3',
    'phone_3_dnc',
    'address',
    'city',
    'state',
    'zip_code',
    'zip4',
    'fips_state_code',
    'fips_county_code',
    'county_name',
    'latitude',
    'longitude',
    'age',
    'gender',
    'address_type',
    'cbsa',
    'census_tract',
    'census_block_group',
    'census_block',
    'scf',
    'dma',
    'msa',
    'congressional_district',
    'head_of_household',
    'birth_month_and_year',
    'prop_type',
    'n_household_children',
    'credit_range',
    'household_income',
    'household_net_worth',
    'home_owner_status',
    'marital_status',
    'occupation',
    'median_home_value',
    'education',
    'length_of_residence',
    'n_household_adults',
    'political_party',
    'health_beauty_products',
    'cosmetics',
    'jewelry',
    'investment_type',
    'investments',
    'pet_owner',
    'pets_affinity',
    'health_affinity',
    'diet_affinity',
    'fitness_affinity',
    'outdoors_affinity',
    'boating_sailing_affinity',
    'camping_hiking_climbing_affinity',
    'fishing_affinity',
    'hunting_affinity',
    'aerobics',
    'nascar',
    'scuba',
    'weight_lifting',
    'healthy_living_interest',
    'motor_racing',
    'foreign_travel',
    'self_improvement',
    'walking',
    'fitness',
    'ethnicity_detail',
    'ethnic_group',
    'md5',
    'insight',
    'email' # added for normalization
]


# -- Dataframe Functions --
@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV file and return a DataFrame"""
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def normalize_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize multiple emails per lead into a single 'email' field (from the email_1, email_2, email_3 columns) where 
    each email is a separate row.
    
    By default, this will keep leads with no email address.
    If the user wants to exclude leads with no email address, it will be handled in the main app loop.
    """
    email_columns = ['email_1', 'email_2', 'email_3']
    rows = []

    for _, row in df.iterrows():
        emails = []
        for col in email_columns:
            if pd.notna(row[col]):
                emails.append(row[col].strip())
        
        # create new rows, drop original email columns
        for email in emails:
            new_row = row.drop(email_columns)
            new_row['email'] = email
            rows.append(new_row)
        
        # include leads with no email address
        if not emails:
            new_row = row.drop(email_columns)
            new_row['email'] = None
            rows.append(new_row)
    return pd.DataFrame(rows)


# -- Mailchimp Client Functions --
@st.cache_resource
def get_mailchimp_client(api_key: str, server_prefix: str) -> Client:
    """Initialize and return a Mailchimp client"""
    client = Client()
    client.set_config({"api_key": api_key, "server": server_prefix})
    return client

@st.cache_data
def verify_mailchimp_credentials(api_key: str, server_prefix: str) -> bool:
    """Verify the users Mailchimp API credentials"""
    client = get_mailchimp_client(api_key, server_prefix)
    try:
        client.ping.get()
        return True
    except ApiClientError as error:
        st.error(f"Error verifying Mailchimp credentials: {error.text}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return False
    
@st.cache_data
def fetch_mailchimp_lists(api_key: str, server_prefix: str) -> dict[str, Any]:
    return get_mailchimp_client(api_key, server_prefix).lists.get_all_lists()


def send_to_mailchimp(df: pd.DataFrame, api_key: str, server_prefix: str, list_id: str, status: str) -> Tuple[int, list[Tuple[str, str]]]:
    """
    Send categorized tags to a Mailchimp list.
    
    Note, this will not update leads that already exist (no way to tag them if upserted).
    This will only add new leads to the list.
    
    Returns the number of successful additions and a list of errors.
    """
    def _safe_get(val: Any) -> str:
        """temporary wrapper to handle NaN values"""
        return "" if pd.isna(val) else val

    def _clean_phone(phone: Any) -> str:
        """Clean and format phone number to (###) ### - ####"""
        
        if pd.isna(phone):
            return ""
        
        try:
            phone = str(int(float(phone))).strip()
            if len(phone) == 10:
                return f"({phone[:3]}) {phone[3:6]} - {phone[6:]}"
            return ""
        except Exception as e:
            return ""
        
    successes: int = 0
    errors: list[Tuple[str, str]] = []

    for _, row in df.iterrows():
        try:
            email = row.get("email")
            if not email:
                continue

            merge_fields = {
                "FNAME": _safe_get(row.get("first_name", "")),
                "LNAME": _safe_get(row.get("last_name", "")),
                "PHONE": _clean_phone(row.get("phone_1", "")),
                "BIRTHDAY": _safe_get(row.get("birth_month_and_year", "")),
                "ADDRESS": {
                    "addr1": _safe_get(row.get("address", "")),
                    "city": _safe_get(row.get("city", "")),
                    "state": _safe_get(row.get("state", "")),
                    "zip": str(_safe_get(row.get("zip_code", ""))),
                    "country": "USA",
                }
            }
                        
            payload = {
                "email_address": email,
                "status": status,
                "merge_fields": merge_fields,
            }
                                    
            if "tags" in df.columns:
                raw_tags = _safe_get(row.get("tags", ""))
                raw_tags = raw_tags.split(",") if isinstance(raw_tags, str) else []
                if raw_tags:
                    payload["tags"] = [tag.strip() for tag in raw_tags if tag.strip()]

            url = f"https://{server_prefix}.api.mailchimp.com/3.0/lists/{list_id}/members"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Basic " + base64.b64encode(f"anystring:{api_key}".encode()).decode(),
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload))

            if response.status_code >= 400:
                errors.append((email, response.text))
            else:
                successes += 1

        except Exception as e:
            errors.append((email, str(e)))
    
    return successes, errors


# -- Tagging Functions --
@st.cache_data
def tag_leads(df: pd.DataFrame, tag_mapping: dict[str, list[str]], tagger_cls: BaseTagger = StandardTagger, priority_list: list[str] | None = None) -> pd.DataFrame:
    tagger = tagger_cls(df, tag_mapping, priority_list)
    return tagger.apply_tags()


# -- Main App Loop --
st.title("Real Intent's Mailchimp Tool")

st.markdown("""
    <div border-radius: 5px;">
        <p>You can either:</p>
        <ul>
            <li><strong>Download a Mailchimp compatible CSV file</strong> for manual upload to Mailchimp.</li>
            <li><strong>Send leads directly to the Mailchimp API</strong> by mapping intent categories to tags that you choose. This will allow for automatic tagging within a list in your Mailchimp account.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

api_key = st.text_input("Enter your Mailchimp API Key:", type="password")
server_prefix = st.text_input("Enter your Mailchimp Server Prefix (e.g. us7):")

mailchimp_ready = False

if api_key and server_prefix:
    mailchimp_ready = verify_mailchimp_credentials(api_key, server_prefix)

if mailchimp_ready:
    st.success("Mailchimp credentials verified successfully!")
        
uploaded_file = st.file_uploader("Upload Your Real Intent CSV", type="csv")

if uploaded_file:
    df = load_csv(uploaded_file)
    
    required_columns = ["email_1", "email_2", "email_3", "first_name", "last_name"] # minimum required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if not all(col in df.columns for col in required_columns):
        st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}. Currently missing: {', '.join(missing_columns)}")
        st.stop()
    
    df = normalize_emails(df)
    
    include_no_email = st.checkbox("Include leads with no email address", value=True)
    
    if not include_no_email:
        df = df.dropna(subset=["email"])

    # hoist email, name to first columns
    df = df[["email", "first_name", "last_name"] + [col for col in df.columns if col not in ["email", "first_name", "last_name"]]]

    user_choice = st.radio("What would you like to do?", ["Download CSV file", "Send to Mailchimp"])

    if user_choice == "Download CSV file":
        st.subheader("Mailchimp Compatible CSV")
        st.write(df)
        st.write("This CSV contains one row per email address, for uploading to Mailchimp.")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV file", csv, "real-intent-mailchimp-leads.csv", "text/csv")
    
    elif user_choice == "Send to Mailchimp" and mailchimp_ready:
        lists_data = fetch_mailchimp_lists(api_key, server_prefix)
        if lists_data:
            audience_options = {lst['name']: lst['id'] for lst in lists_data['lists']}
            list_name = st.selectbox("Select a List", list(audience_options.keys()))
            list_id = audience_options[list_name]
        else:
            st.error("No lists found in your Mailchimp account.")
            st.stop()

        st.write("""
            Please map the intent columns to tags.
            Separate multiple tags with commas. For example: Buyer, Seller, Investor, etc...
        """)
        
        intent_columns = [col for col in df.columns if col not in default_columns]

        tag_mapping: dict[str, list[str]] = {}
        
        for col in intent_columns:
            tags_input = st.text_input(f"Map '{col}' intent column to a Mailchimp tag(s)")
            if tags_input:
                tags_list = [tag.strip() for tag in tags_input.split(",")]
                tag_mapping[col] = tags_list
        
        tagging_options = st.radio("Tagging Options", ["Standard Tagger", "Custom Tagger #1"], index=0)
        
        st.write(f"{tagging_options}: {BaseTagger.get_description(tagging_options)}")
                
        if tagging_options == "Standard Tagger":
            tagged_df = tag_leads(df, tag_mapping, StandardTagger)
        elif tagging_options == "Custom Tagger #1":
            
            # Grab unique tags
            all_tags = sorted({tag for tags in tag_mapping.values() for tag in tags})
           
            st.subheader("Set Priority for Tags")
            st.caption("Drag to reorder tags. The first tag has the highest priority, the second tag has the second highest priority, and so on.")
            
            sorted_priority = sort_items(all_tags)

            # only occurs if there are no tags
            if not sorted_priority:
                st.warning("Please input at least one mapping to continue.")
                st.stop()
            
            tagged_df = tag_leads(df, tag_mapping, Custom_1_Tagger, sorted_priority)


        # hoist email, tags, name to first columns
        tagged_df = tagged_df[["email", "tags", "first_name", "last_name"] + [col for col in tagged_df.columns if col not in ["email", "tags", "first_name", "last_name"]]]

        st.subheader("Tagged Leads")
        st.write(tagged_df)
        
        st.write("This CSV contains one row per email address, with tags assigned based on the categories you mapped. Note uploading this to Mailchimp will NOT add these tags to the leads. However, you can still choose to download this CSV.")
        csv_categorized = tagged_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_categorized, "real-intent-mailchimp-leads-tagged.csv", "text/csv")

        st.subheader("Subscription Status")
        status_choice = st.selectbox(
            "Choose subscription status for the contacts:",
            options=["", "subscribed", "unsubscribed", "cleaned", "pending", "transactional"],
            index=0,
            help="Subscriber's current status. (required)",
        )
        
        if status_choice == "":
            st.warning("Please select a subscription status to continue.")
        else:
            if st.button("Confirm Tags/List and Send to Mailchimp"):
                with st.spinner("Sending leads to Mailchimp..."):
                    successes, errors = send_to_mailchimp(tagged_df, api_key, server_prefix, list_id, status_choice)
                
                st.success(f"Successfully sent {successes} leads to Mailchimp.")
                if errors:
                    st.error("Errors occurred while sending leads to Mailchimp:")
                    for email, error in errors:
                        st.write(f"Email: {email}, Error: {error}")
                
    elif user_choice == "Send to Mailchimp" and not mailchimp_ready:
        st.warning("Please enter your Mailchimp API Key and Server Prefix to send leads.")