import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

def get_news_categories(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    categories = []

    # print(soup.prettify())
    # Extract news categories using the specified class name
    category_elements = soup.find_all('li', class_='nav-item')
    for category_element in category_elements:
        category_name = category_element.text.strip()
        category_url = category_element.find('a')['href']
        categories.append({'name': category_name, 'url': category_url})

    return categories

def get_links(news_categories,main_link,categories):
    links_with_text = []
    seen_links = set()

    for category in news_categories:
        url = category['url']
        category_name = category['name']

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract links and their corresponding text
        for link in tqdm(soup.find_all('a', href=True), desc=f"Processing {category_name}"):
            for cat in categories:
                if link['href'].startswith(cat):
                    link_url = link['href']
                    link_text = link.text.strip()
                    if link_url not in seen_links and link_text:
                        data = extract_content(main_link+link_url)
                        links_with_text.append({'url': main_link+link_url, 'title': link_text, 'paragraph':data,'category': category_name})
                        seen_links.add(link_url)
    return links_with_text

def extract_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # print(soup.prettify())
    # Find the meta tag with name="description" and extract its content attribute
    meta_tag = soup.find('meta', attrs={'name': 'description'})

    if meta_tag:
        description_content = meta_tag.get('content')
    else:
        description_content = "Meta tag with name='description' not found."
    return description_content

if __name__ == '__main__':

    url = 'https://www.ekantipur.com/'
    news_url = 'https://www.ekantipur.com/news'
    links_with_text = []
    seen_links = set()

    if os.path.exists('categories.json'):
        news_categories = get_news_categories(url)
        with open('categories.json', 'w') as json_file:
            json.dump(news_categories, json_file, ensure_ascii=False, indent=4)
    else:
        with open('categories.json', 'r') as json_file:
            news_categories = json.load(json_file)

    print("News Categories on ekantipur.com:")
    for category in news_categories:
        print(f"Category Name: {category['name']}")
        print(f"Category URL: {category['url']}")

    categories=['/news','/business','/opinion','/sports','/national','/entertainment','/world','/Education']
    response = requests.get(news_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    for link in soup.find_all('a', href=True):
        if link['href'].startswith('/news'):
            link_url = link['href']
            link_text = link.text.strip()
            if link_url not in seen_links and link_text:
                links_with_text.append({'url': link_url, 'text': link_text})
                seen_links.add(link_url)

    # print extracted links and their corresponding text
    for link_info in links_with_text:
        print("URL:", link_info['url'])
        print("Text:", link_info['text'])
        print("---")

    df = get_links(news_categories=news_categories,main_link=news_url,categories=categories)
    df = pd.DataFrame(df)
    df.to_csv('output.csv', index=False)
    print("CSV file exported successfully.")
