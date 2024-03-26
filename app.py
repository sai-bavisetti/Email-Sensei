import streamlit as st
import pandas as pd
import time


# Replace this with your actual Python code that generates the DataFrame
def generate_dataframe():
  # Your logic to create the DataFrame
  data = {"col1": [1, 2, 3], "col2": ["A", "B", "C"]}
  df = pd.DataFrame(data)
  return df



def main():
  """Streamlit app to display DataFrame on button click."""
  # Define theme colors
  theme = {
    "backgroundColor": "#f0f2f5",  # Adjust background color as desired
    "primaryColor": "#2196f3",  # Adjust primary color as desired
    "textColor": "#333",  # Adjust text color as desired
    # Add more color options as needed (see Streamlit docs)
  }

  st.set_theme(theme)
  
  st.title(":blue[Email Sensei]")
  st.caption(":green[Please send an email to sai_bavisetti@outlook.com]")

  if st.button("Process Emails"):
    try:
      with st.container(height = 700):
        with st.spinner('Wait for it...'):
            time.sleep(5)
        df = generate_dataframe()
        st.success("Check Your Email!")
        st.dataframe(df)
        st.text_area("Value", df.loc[0, 'col2'])
        st.text_area("Value2", df.loc[1, 'col2'])
        st.text_area("Value3", df.loc[2, 'col2'])
        st.download_button(label="Download data as CSV",data=df.to_csv(),file_name='Email_summary.csv',mime='text/csv')
    except Exception as e:
      st.error(f"Error executing: {e}")

if __name__ == "__main__":
  main()
