# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 09:13:28 2025

@author: EG283299
"""

from bs4 import BeautifulSoup
import requests
import pypub
import os

links_list = []
path=r'file:\\\C:\Users\EG283299\Documents\new_treillis\treillis_git\treillis_master\docs\build\epub\\'
for file in os.listdir(path[8:]):
    if file[-4:]=='html':
        links_list.append(path+file)


my_epub = pypub.Epub("treillis")

# def remove_tags(html):
#     soup = BeautifulSoup(html, "html.parser")
#     for data in soup(['li']):
#         data.decompose()

#     return ' '.join(soup.strings)

# for i in range(0, len(links_list)):
#     page = requests.get(links_list[i])
#     chapter_html = remove_tags(page.content)
#     chapter = pypub.create_chapter_from_html(chapter_html, title=f"Chapter {i + 1}")
#     my_epub.add_chapter(chapter)

for i in range(len(links_list)):
    file = links_list[i]
    chapter = pypub.create_chapter_from_url(file)
    my_epub.add_chapter(chapter)
my_epub.create(path[8:]+"/My Epub.epub")
