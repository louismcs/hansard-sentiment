""" File containing the MatchException and Database classes """


import csv
import sqlite3


class MatchException(Exception):
    """ Raised when there is no appropriate match in the database for a given speaker """
    pass

class Database:
    """ Stores the corpus of MPs votes and speeches and defines the operations that can be
                                   performed for input to and output from the database """

    def __init__(self):
        self.conn = None
        self.curs = None


    def __enter__(self):
        self.conn = sqlite3.connect('/Data/Iraq/corpus.db')
        self.curs = self.conn.cursor()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


    def generate_name_list(self):
        """ Generates a python dictionary mapping mp names to ids """
        self.curs.execute("SELECT ID, FULL_NAME, GIVEN_NAME, FAMILY_NAME FROM MEMBER")
        rows = self.curs.fetchall()
        return rows


    def insert_debate(self, url, day, title):
        """ Inserts a debate into the database given its data """
        try:
            self.curs.execute("INSERT INTO DEBATE (URL, DATE, TITLE) VALUES (?, ?, ?)",
                              (url, day.strftime('%Y-%m-%d'), title))
            self.conn.commit()
        except sqlite3.OperationalError:
            print('FAILED DEBATE INSERT: {} - {} - {}'.format(url, day.strftime('%Y-%m-%d'), title))


    def insert_speech(self, url, member_id, quote):
        """ Inserts a speech into the database given its data """
        try:
            self.curs.execute('''INSERT INTO SPEECH (DEBATE_URL, MEMBER_ID, QUOTE)
                                        VALUES (?, ?, ?)''', (url, member_id, quote))
            self.conn.commit()
        except sqlite3.OperationalError:
            print('FAILED SPEECH INSERT: {} - {} - {}'.format(url, member_id, quote))


    def create_tables(self):
        """ Creates the Member, Division and Vote tables """

        self.curs.execute('''CREATE TABLE MEMBER
                (ID                TEXT   PRIMARY KEY   NOT NULL,
                FULL_NAME         TEXT,
                GIVEN_NAME        TEXT,
                ADDITIONAL_NAME   TEXT,
                FAMILY_NAME       TEXT,
                PARTY             TEXT,
                CONSTITUENCY      TEXT);''')
        self.conn.commit()

        self.curs.execute('''CREATE TABLE DIVISION
                (ID      TEXT   PRIMARY KEY   NOT NULL,
                DATE    TEXT,
                TITLE   TEXT);''')
        self.conn.commit()

        self.curs.execute('''CREATE TABLE VOTE
                (MEMBER_ID     TEXT   NOT NULL,
                DIVISION_ID   TEXT   NOT NULL,
                VOTE          TEXT,
                PRIMARY KEY(MEMBER_ID, DIVISION_ID),
                FOREIGN KEY(MEMBER_ID)   REFERENCES MEMBER(ID),
                FOREIGN KEY(DIVISION_ID) REFERENCES DIVISION(ID));''')
        self.conn.commit()

        self.curs.execute('''CREATE TABLE DEBATE
                (URL     TEXT   PRIMARY KEY   NOT NULL,
                DATE    TEXT,
                TITLE   TEXT);''')
        self.conn.commit()

        self.curs.execute('''CREATE TABLE SPEECH
                (DEBATE_URL   TEXT,
                MEMBER_ID    TEXT,
                QUOTE        TEXT,
                FOREIGN KEY(DEBATE_URL) REFERENCES DEBATE(URL),
                FOREIGN KEY(MEMBER_ID)   REFERENCES MEMBER(ID));''')
        self.conn.commit()


    def insert_division(self, division_id, division_date, title):
        """ Inserts a division into the database given its data """
        try:
            self.curs.execute("INSERT INTO DIVISION (ID, DATE, TITLE) VALUES (?, ?, ?)",
                              (division_id, division_date, title))
            self.conn.commit()
        except sqlite3.OperationalError:
            print('FAILED DIVISION INSERT: {} - {} - {}'.format(division_id, division_date, title))


    def insert_member(self, member_id, member_data):
        """ Inserts a member into the database given their data """
        try:
            self.curs.execute('''INSERT INTO MEMBER
                            (ID, FULL_NAME, GIVEN_NAME, ADDITIONAL_NAME, FAMILY_NAME, PARTY, CONSTITUENCY)
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                              (member_id, member_data['full_name'],
                               member_data['given_name'], member_data['additional_name'],
                               member_data['family_name'], member_data['party'],
                               member_data['constituency']))
            self.conn.commit()
        except sqlite3.OperationalError:
            print('FAILED MEMBER INSERT: {}'.format(member_data['full_name']))


    def insert_vote(self, member_id, division_id, member_vote):
        """ Inserts a vote into the database given its data """
        try:
            self.curs.execute("INSERT INTO VOTE (MEMBER_ID, DIVISION_ID, VOTE) VALUES (?, ?, ?)",
                              (member_id, division_id, member_vote))
            self.conn.commit()
        except sqlite3.OperationalError:
            print('FAILED VOTE INSERT: {} - {} - {}'
                  .format(member_id, division_id, member_vote))
        except sqlite3.IntegrityError:
            print('FAILED VOTE INSERT (DUPLICATE): {} - {} - {}'
                  .format(member_id, division_id, member_vote))


    def get_all_division_ids(self):
        """ Returns the ids of all divisions in the database """

        self.curs.execute("SELECT ID FROM DIVISION")
        return self.curs.fetchall()


    def generate_csv(self, table):
        """ Outputs a csv for the given table """
        data = self.curs.execute("SELECT * FROM " + table)
        filename = table.lower() + '.csv'
        csv_file = open(filename, 'w', newline="")
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerows(data)
        csv_file.close()


    def get_debates_from_term(self, term):
        """ Returns a list of debate ids where the term is in the debate title """

        self.curs.execute('''SELECT URL FROM DEBATE
                        WHERE TITLE LIKE ? COLLATE NOCASE''', ('%{}%'.format(term),))

        rows = self.curs.fetchall()

        return [row[0] for row in rows]


    def get_debates(self, settings):
        """ Returns a list of debate ids matching the given settings """

        if settings['all_debates']:
            self.curs.execute("SELECT URL FROM DEBATE")
            rows = self.curs.fetchall()
            ret = [row[0] for row in rows]
        else:
            debates = set()
            for term in settings['debate_terms']:
                debates = debates.union(set(self.get_debates_from_term(term)))
            ret = list(debates)

        return ret


    def get_all_members_from_term(self, term, division_id):
        """ Returns a list of member ids corresponding to members who voted in a
            given division and spoke in a debate matching a given term """

        self.curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                            WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                                WHERE TITLE LIKE ? COLLATE NOCASE)
                            AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                            VOTE.MEMBER_ID = MEMBER.ID
                                            WHERE VOTE.DIVISION_ID=? AND
                                            (VOTE.VOTE='AyeVote' OR VOTE.VOTE='NoVote')) ''',
                          ('%{}%'.format(term), division_id))

        rows = self.curs.fetchall()

        return [row[0] for row in rows]


    def get_aye_members_from_term(self, term, division_id):
        """ Returns a list of member ids corresponding to members who voted aye
            in a given division and spoke in a debate matching a given term """

        self.curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                            WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                                WHERE TITLE LIKE ? COLLATE NOCASE)
                            AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                            VOTE.MEMBER_ID = MEMBER.ID
                                            WHERE VOTE.DIVISION_ID=? AND
                                            (VOTE.VOTE='AyeVote')) ''',
                          ('%{}%'.format(term), division_id))

        rows = self.curs.fetchall()

        return [row[0] for row in rows]


    def get_no_members_from_term(self, term, division_id):
        """ Returns a list of member ids corresponding to members who voted no
            in a given division and spoke in a debate matching a given term """

        self.curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                            WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                                WHERE TITLE LIKE ? COLLATE NOCASE)
                            AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                            VOTE.MEMBER_ID = MEMBER.ID
                                            WHERE VOTE.DIVISION_ID=? AND
                                            (VOTE.VOTE='NoVote')) ''',
                          ('%{}%'.format(term), division_id))

        rows = self.curs.fetchall()

        return [row[0] for row in rows]


    def get_member_no_of_speeches(self, debate_ids, member_id):
        """ Returns the number of speeches made by the given member in the given debates """

        statement = ''' SELECT COUNT(*) FROM SPEECH WHERE DEBATE_URL IN ({debates})
                        AND MEMBER_ID={member} '''.format(
                            debates=','.join(['?']*len(debate_ids)),
                            member=member_id)

        self.curs.execute(statement, debate_ids)

        return self.curs.fetchone()[0]


    def get_speech_texts(self, member, debate):
        """ Returns a list of strings of the speeches of a given MP in a given debate """

        self.curs.execute('''SELECT QUOTE FROM SPEECH
                        WHERE MEMBER_ID=? AND DEBATE_URL=?''', (member['id'], debate))

        rows = self.curs.fetchall()

        return [{'text': row[0], 'votes': member['votes'], 'member': member['id']} for row in rows]


    def is_aye_vote(self, division_id, member_id):
        """ Returns a boolean value of whether the given
            member voted 'Aye' in the given division """

        self.curs.execute('''SELECT VOTE FROM VOTE WHERE MEMBER_ID=? AND DIVISION_ID=? ''',
                          (member_id, division_id))

        return self.curs.fetchone()[0] == 'AyeVote'


    def get_members_from_term(self, term, division_id):
        """ Returns a list of member ids corresponding to members who voted in a
            given division and spoke in a debate matching a given term """

        self.curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                            WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                                WHERE TITLE LIKE ? COLLATE NOCASE)
                            AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                            VOTE.MEMBER_ID = MEMBER.ID
                                            WHERE VOTE.DIVISION_ID=? AND
                                            (VOTE.VOTE='AyeVote' OR VOTE.VOTE='NoVote')) ''',
                          ('%{}%'.format(term), division_id))

        rows = self.curs.fetchall()

        return [row[0] for row in rows]


    def get_number_of_speeches(self, debate_ids, member_ids):
        """ Returns the total number of speeches in the given debates by the given members """

        statement = ''' SELECT COUNT(*) FROM SPEECH WHERE DEBATE_URL IN ({debates})
                        AND MEMBER_ID IN ({members}) '''.format(
                            debates=','.join(['?']*len(debate_ids)),
                            members=','.join(['?']*len(member_ids)))

        self.curs.execute(statement, debate_ids + member_ids)

        return self.curs.fetchone()[0]
