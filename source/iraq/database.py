import csv
import sqlite3


DB_PATH = '../database/corpus.db'


class MatchException(Exception):
    """ Raised when there is no appropriate match in the database for a given speaker """
    pass


def generate_name_list(curs):
    """ Generates a python dictionary mapping mp names to ids """
    curs.execute("SELECT ID, FULL_NAME, GIVEN_NAME, FAMILY_NAME FROM MEMBER")
    rows = curs.fetchall()
    return rows


def insert_debate(database, url, day, title):
    """ Inserts a debate into the database given its data """
    try:
        database['curs'].execute("INSERT INTO DEBATE (URL, DATE, TITLE) VALUES (?, ?, ?)",
                                 (url, day.strftime('%Y-%m-%d'), title))
        database['conn'].commit()
    except sqlite3.OperationalError:
        print('FAILED DEBATE INSERT: {} - {} - {}'.format(url, day.strftime('%Y-%m-%d'), title))


def insert_speech(database, url, member_id, quote):
    """ Inserts a speech into the database given its data """
    try:
        database['curs'].execute('''INSERT INTO SPEECH (DEBATE_URL, MEMBER_ID, QUOTE)
                                    VALUES (?, ?, ?)''', (url, member_id, quote))
        database['conn'].commit()
    except sqlite3.OperationalError:
        print('FAILED SPEECH INSERT: {} - {} - {}'.format(url, member_id, quote))


def make_connection():
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()

    return {
        'conn': conn,
        'curs': curs,
    }


def close_connection(conn):
    conn.close()


def create_tables():
    """ Creates the Member, Division and Vote tables """
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()

    curs.execute('''CREATE TABLE MEMBER
            (ID                TEXT   PRIMARY KEY   NOT NULL,
             FULL_NAME         TEXT,
             GIVEN_NAME        TEXT,
             ADDITIONAL_NAME   TEXT,
             FAMILY_NAME       TEXT,
             PARTY             TEXT,
             CONSTITUENCY      TEXT);''')
    conn.commit()

    curs.execute('''CREATE TABLE DIVISION
            (ID      TEXT   PRIMARY KEY   NOT NULL,
             DATE    TEXT,
             TITLE   TEXT);''')
    conn.commit()

    curs.execute('''CREATE TABLE VOTE
            (MEMBER_ID     TEXT   NOT NULL,
             DIVISION_ID   TEXT   NOT NULL,
             VOTE          TEXT,
             PRIMARY KEY(MEMBER_ID, DIVISION_ID),
             FOREIGN KEY(MEMBER_ID)   REFERENCES MEMBER(ID),
             FOREIGN KEY(DIVISION_ID) REFERENCES DIVISION(ID));''')
    conn.commit()

    curs.execute('''CREATE TABLE DEBATE
            (URL     TEXT   PRIMARY KEY   NOT NULL,
             DATE    TEXT,
             TITLE   TEXT);''')
    conn.commit()

    curs.execute('''CREATE TABLE SPEECH
            (DEBATE_URL   TEXT,
             MEMBER_ID    TEXT,
             QUOTE        TEXT,
             FOREIGN KEY(DEBATE_URL) REFERENCES DEBATE(URL),
             FOREIGN KEY(MEMBER_ID)   REFERENCES MEMBER(ID));''')
    conn.commit()

    conn.close()


def insert_division(conn, curs, division_id, division_date, title):
    """ Inserts a debate into the database given its data """
    try:
        curs.execute("INSERT INTO DIVISION (ID, DATE, TITLE) VALUES (?, ?, ?)",
                     (division_id, division_date, title))
        conn.commit()
    except sqlite3.OperationalError:
        print('FAILED DIVISION INSERT: {} - {} - {}'.format(division_id, division_date, title))


def insert_member(conn, curs, member_id, member_data):
    """ Inserts a member into the database given their data """
    try:
        curs.execute('''INSERT INTO MEMBER
                        (ID, FULL_NAME, GIVEN_NAME, ADDITIONAL_NAME, FAMILY_NAME, PARTY, CONSTITUENCY)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (member_id, member_data['full_name'],
                      member_data['given_name'], member_data['additional_name'],
                      member_data['family_name'], member_data['party'],
                      member_data['constituency']))
        conn.commit()
    except sqlite3.OperationalError:
        print('FAILED MEMBER INSERT: {}'.format(member_data['full_name']))


def insert_vote(conn, curs, member_id, division_id, member_vote):
    """ Inserts a vote into the database given its data """
    try:
        curs.execute("INSERT INTO VOTE (MEMBER_ID, DIVISION_ID, VOTE) VALUES (?, ?, ?)",
                     (member_id, division_id, member_vote))
        conn.commit()
    except sqlite3.OperationalError:
        print('FAILED VOTE INSERT: {} - {} - {}'
              .format(member_id, division_id, member_vote))
    except sqlite3.IntegrityError:
        print('FAILED VOTE INSERT (DUPLICATE): {} - {} - {}'
              .format(member_id, division_id, member_vote))


def get_all_division_ids(curs):
    curs.execute("SELECT ID FROM DIVISION")
    return curs.fetchall()

def generate_csv(table):
    """ Outputs a csv for the given table """
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()
    data = curs.execute("SELECT * FROM " + table)
    filename = table.lower() + '.csv'
    csv_file = open(filename, 'w', newline="")
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerows(data)
    csv_file.close()


def generate_divisions_csv():
    """ Generates a csv of the contents of the divisions table """
    generate_csv('DIVISION')


def generate_members_csv():
    """ Generates a csv of the contents of the members table """
    generate_csv('MEMBER')


def generate_votes_csv():
    """ Generates a csv of the contents of the votes table """
    generate_csv('VOTE')


def generate_debates_csv():
    """ Generates a csv of the contents of the debates table """
    generate_csv('DEBATE')
