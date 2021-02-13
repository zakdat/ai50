import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # create empty dictionary
    probability_dict = {}

    # check whether page has links
    if corpus[page]:
        # for every link in corpus, copy to dict and set base random probability
        for link in corpus:
            probability_dict[link] = (1 - damping_factor) / len(corpus)
        # with P damping factor, should choose linked page
            if link in corpus[page]:
                probability_dict[link] += damping_factor / len(corpus[page])
    # if no linked pages, equal probability of any page
    else:
        for link in corpus:
            probability_dict[link] = 1 / len(corpus)

    # return dictionary, keys = pages, values = probabilities (sum to 1)
    return probability_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create empty dict
    pagerank_dict = {}

    # set all values to 0
    for page in corpus:
        pagerank_dict[page] = 0

    # generate first sample choosing page at random
    sample = random.choice(list(corpus))
    # sample n pages
    for _ in range(n):
        # add sample to dict and update sample
        pagerank_dict[sample] += (1 / n)    
        # pass sample to transition model function to generate transition model dict
        transition_dict = transition_model(corpus, sample, damping_factor)
        # choose new sample with weighted probability
        sample = random.choices(list(transition_dict.keys()), weights = transition_dict.values(), k=1)[0]

    # return dict, keys = pagenames, values = pageranks. sum to 1
    return pagerank_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create empty dict
    iterate_dict = {}
    new_dict = {}

    size = len(corpus)

    # assign each page default rank 1/n
    for page in corpus:
        iterate_dict[page] = (1 / size)
    
    # repeat until no value changes more than 0.001
    repeat = True

    while repeat:
        for page in iterate_dict:
            rank_value = float(0)

            for possible_page in corpus:
                # consider each page that links to current page
                if page in corpus[possible_page]:
                    rank_value += iterate_dict[possible_page] / len(corpus[possible_page])
                # A page that has no links is interpreted as having one link for every page (including itself)
                if not corpus[possible_page]:
                    rank_value += iterate_dict[possible_page] / len(corpus)

            new_dict[page] = (1 - damping_factor) / size + damping_factor * rank_value
        
        # check values change more than 0.001
        repeat = False

        for page in iterate_dict:
            if not math.isclose(new_dict[page], iterate_dict[page], abs_tol=0.001):
                repeat = True
            iterate_dict[page] = new_dict[page]

    # return dict with key = page, value = pagerank. sum to 1
    return iterate_dict


if __name__ == "__main__":
    main()
