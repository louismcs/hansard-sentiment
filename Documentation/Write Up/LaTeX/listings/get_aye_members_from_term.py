def get_aye_members_from_term(self, term, division_id):
    ''' Returns a list of member ids corresponding to members who voted aye
        in a given division and spoke in a debate matching a given term '''

    self.__curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                        WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                            WHERE TITLE LIKE ? COLLATE NOCASE)
                        AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                          VOTE.MEMBER_ID = MEMBER.ID
                                          WHERE VOTE.DIVISION_ID=? AND
                                          (VOTE.VOTE='AyeVote')) ''',
                        ('%{}%'.format(term), division_id))

    rows = self.__curs.fetchall()

    return [row[0] for row in rows]