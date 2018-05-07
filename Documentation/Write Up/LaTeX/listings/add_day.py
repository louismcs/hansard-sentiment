def add_day(corpus, day, member_lists):
    '''Gets the speeches for a given day'''
    date_string = day.strftime('%Y/%b/%d').lower()
    url = 'http://hansard.millbanksystems.com/sittings/{}.js'.format(date_string)
    res = requests.get(url)

    try:
        obj = json.loads(res.text)
        try:
            sections = obj[0]['house_of_commons_sitting']['top_level_sections']
            for section in sections:
                try:
                    sec = section['section']
                    add_debate(corpus, 'http://hansard.millbanksystems.com/commons/{}/{}'
                               .format(date_string, sec['slug']), day, sec['title'],
                               member_lists)
                except KeyError:
                    pass
        except KeyError:
            pass

    except ValueError:
        print('No data for {}'.format(date_string))