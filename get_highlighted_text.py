def get_response(text):
    """
    this function generates the final sentence after changing the format of the highlighted text from ElasticSearch.
    :param text: input text
    :return:
    """
    from nltk.corpus import stopwords
    import requests

    stopword_list = list(set(stopwords.words("english")))
    url = "http://192.168.34.9:5000/api/get_highlights"  # the URL of the ElasticSearch host goes here
    data = {"text": text}
    try:
        response = requests.post(url=url, data=data)
        response_json = response.json()
        highlighted_words = []

        qb_info = response_json["qb"]
        wiki_info = response_json["wiki"]

        for info in wiki_info:
            info_split = info.split()
            highlighted_words += [word.replace("<em>", "").replace("</em>", "") for word in info_split if
                                  "<em>" in word]

        for info in qb_info:
            info_split = info.split()
            highlighted_words += [word.replace("<em>", "").replace("</em>", "") for word in info_split if
                                  "<em>" in word]

        original_highlighted_words = list(set(highlighted_words) - set(stopword_list))
        highlighted_words = list(set([word.lower() for word in original_highlighted_words]) - set(stopword_list))

        final_text = []
        final_sentences = []
        sentences = text.split(".")
        for sentence in sentences:
            prev_list = []
            for word in sentence.split():
                if word.lower() in highlighted_words:
                    word = word.upper()
                    if prev_list and prev_list[-1]:
                        elem = final_text.pop()
                        prev_list.pop()
                        word = elem + "_" + word

                    prev_list.append(True)

                else:
                    prev_list.append(False)
                final_text.append(word)
            reconstructed_sentence = " ".join(final_text) + "."
            final_text = []
            final_sentences.append(reconstructed_sentence)

        return "".join(final_sentences)

    except Exception as err:
        print(err)

    return text


"""
below are the driver functions to independently test this functionality
def main():
    text = "Book about harry potter. This was written by JK Rowling"
    final_text = get_response(text)
    print(final_text)


if __name__ == "__main__":
    main()
"""
