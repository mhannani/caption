import streamlit as st


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(icon_names, links):
    st.markdown(f'<a href="{links[0]}" title="Github Repository"><i class="{icon_names[0]}"></i></a>'
                f' <a href="{links[1]}" title="Web Application"><i class="{icon_names[1]}"></i></a>',
                unsafe_allow_html=True)


