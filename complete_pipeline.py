import requests
from bs4 import BeautifulSoup
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
import time

def collect_book_urls(base_url, num_pages, retries=3, delay=5):
    """
    Collects book URLs from the "Books to Scrape" website with error handling.

    Parameters:
    base_url (str): The base URL of the website to scrape.
    num_pages (int): The number of pages to scrape.
    retries (int): The number of retries for each request in case of failure.
    delay (int): The delay (in seconds) between retries.

    Returns:
    urls (list): A list of collected book URLs.
    """
    urls = []
    for i in range(1, num_pages + 1):
        for attempt in range(retries):
            try:
                response = requests.get(f"{base_url}/catalogue/page-{i}.html")
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                for link in soup.select('h3 > a'):
                    urls.append(base_url + '/catalogue/' + link['href'])
                break  # Break out of the retry loop if successful
            except requests.RequestException as e:
                print(f"Failed to retrieve page {i} (attempt {attempt + 1}): {e}")
                time.sleep(delay)  # Wait before retrying
    return urls

def clustering_step(urls, training_size, eps=1.0, min_samples=2):
    """
    The Clustering Step of TC algorithm using DBSCAN to select URLs for training data.

    Parameters:
    urls (list): URLs downloaded from several web pages on a website.
    training_size (int): Desired number of URLs for the annotation step.
    eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    selected_urls (list): Recommended URLs for training data.
    """
    selected_urls = []
    
    # Convert URLs to numeric vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(urls)
    
    # Apply DBSCAN clustering
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    print(f"Clusters: {clusters}")
    
    count_clusters = []

    for i in range(len(set(clusters))):
        cluster_i = [urls[j] for j in range(len(urls)) if clusters[j] == i]
        print(f"Cluster {i}: {cluster_i}")
        if cluster_i:  # Check if the cluster is not empty
            selected_urls.append(cluster_i.pop())
            count_clusters.append(len(cluster_i))
            if training_size == len(selected_urls):
                break

    rest_training_size = training_size - len(selected_urls)
    if rest_training_size > 0:
        for i in range(len(count_clusters)):
            count_clusters[i] = int(count_clusters[i] * rest_training_size / len(urls))

        for i in range(len(count_clusters)):
            for _ in range(count_clusters[i]):
                cluster_i = [urls[j] for j in range(len(urls)) if clusters[j] == i]
                if cluster_i:  # Check if the cluster is not empty
                    selected_urls.append(cluster_i.pop())
                    if len(selected_urls) == training_size:
                        break
            if len(selected_urls) == training_size:
                break

    return selected_urls

def download_webpage(url):
    """
    Downloads the webpage content from the given URL.

    Parameters:
    url (str): The URL of the webpage to download.

    Returns:
    str: The content of the downloaded webpage.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def parse_images(webpage_content):
    """
    Parses the images from the webpage content.

    Parameters:
    webpage_content (str): The content of the webpage.

    Returns:
    list: A list of image elements extracted from the webpage.
    """
    soup = BeautifulSoup(webpage_content, 'html.parser')
    images = soup.find_all('img')
    return images

def extract_textual_data(image):
    """
    Extracts the textual data for a given image element.

    Parameters:
    image (Tag): The image element.

    Returns:
    str: The textual data extracted from the image element.
    """
    parent1 = image.find_parent()
    parent2 = parent1.find_parent() if parent1 else None
    textual_data = f"Parent1: {parent1}, Parent2: {parent2}, Img: {image}"
    return textual_data

def annotate_relevant_images(images):
    """
    Simulates user annotation of relevant images.

    Parameters:
    images (list): A list of image elements.

    Returns:
    list: A list of indices of relevant images.
    """
    # For the purpose of this example, we will randomly select relevant images.
    # In a real scenario, this function would involve user interaction.
    relevant_indices = random.sample(range(len(images)), k=min(3, len(images)))
    return relevant_indices

def prepare_training_dataset(urls):
    """
    Prepares the training dataset by annotating relevant images.

    Parameters:
    urls (list): A list of URLs suggested by the clustering step.

    Returns:
    list: The training dataset with annotated relevant images.
    """
    training_dataset = []
    
    for url in urls:
        webpage_content = download_webpage(url)
        images = parse_images(webpage_content)
        
        for image in images:
            textual_data = extract_textual_data(image)
            training_dataset.append((textual_data, 0))
        
        relevant_indices = annotate_relevant_images(images)
        
        for index in relevant_indices:
            training_dataset[index] = (training_dataset[index][0], 1)
    
    return training_dataset

def prepare_feature_vectors(training_dataset):
    """
    Prepares feature vectors from the training dataset.

    Parameters:
    training_dataset (list): The training dataset obtained from the annotation step.

    Returns:
    tuple: A tuple containing the dictionary of tokens and the feature vectors.
    """
    dictionary = []
    feature_vectors = []

    for textual_data, relevance in training_dataset:
        bag_of_tokens = textual_data.split()  # Tokenization
        for token in bag_of_tokens:
            if token not in dictionary:
                dictionary.append(token)

    feature_vectors = np.zeros((len(training_dataset), len(dictionary) + 1))

    for i, (textual_data, relevance) in enumerate(training_dataset):
        bag_of_tokens = textual_data.split()  # Tokenization
        for token in bag_of_tokens:
            pos = dictionary.index(token)
            feature_vectors[i][pos] += 1
        feature_vectors[i][-1] = relevance

    return dictionary, feature_vectors

# Example usage
base_url = "http://books.toscrape.com"
num_pages = 5
training_size = 10

# Collect URLs
collected_urls = collect_book_urls(base_url, num_pages)
print("Collected URLs:", collected_urls)

# Perform clustering step
selected_urls = clustering_step(collected_urls, training_size)
print("Selected URLs for training:", selected_urls)

# Prepare training dataset
training_dataset = prepare_training_dataset(selected_urls)
print("Training Dataset:", training_dataset)

# Prepare feature vectors
dictionary, feature_vectors = prepare_feature_vectors(training_dataset)
print("Dictionary:", dictionary)
print("Feature Vectors:", feature_vectors)
