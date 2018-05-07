def get_paragraph_text(paragraph):
    '''Converts a paragraph tag to plain text'''
    paragraph = re.sub(r'<p.*?>', '', paragraph)
    paragraph = re.sub(r'</p.*?>', '', paragraph)
    paragraph = re.sub(r'<a.*?>.*?</a>', '', paragraph)
    paragraph = re.sub(r'<span.*?>.*?</span>', '', paragraph)
    paragraph = re.sub(r'<q.*?>.*?</q>', '\'', paragraph)
    return paragraph