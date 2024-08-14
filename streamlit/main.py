from streamlit_option_menu import option_menu
from page import page1, page2, page3, page4, page5
import streamlit as st
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

# Set page config
# Streamlit's built-in icons
pages = {
    "Data Generation": {
        "icon": "hdd",
        "func": page1,
    },
    "Data Analysis": {
        "icon": "bar-chart",
        "func": page2,
    },
    "Model Train": {
        "icon": "layers",
        "func": page3,
    },
    "Check Logs": {
        "icon": "file-earmark-text",
        "func": page4,
    },
    "Model Test": {
        "icon": "check-circle",
        "func": page5,
    }
}


if __name__ == "__main__":
    with st.sidebar:
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            default_index=default_index,
        )
    if selected_page in pages:
        pages[selected_page]["func"]()
