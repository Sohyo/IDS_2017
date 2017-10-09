Total = 60 + 7/100

1.1 7/7p

1.2 4/7p
Talk more on how you transform your data in dummy variables

1.3 5/7p
You miss the maximum value that can reach the support of the interaction 

1.4 9/9p


2.1. 8/10p
Very good setup. 

Good explanation of preprocessing. Trimming of data frame appears to have been done outside the code submitted (if this is not the case, please indicate where it was done and this will be revisited).

Very good summary statistics. 
2.2 [8/15]
The alignment in this table should have been fixed. Otherwise, the investigation, argument and conclusion are nice. 
Also, other classical music composers did not come up in your list. Any layman would expect that a person liking Beethoven might be into classical music and might have liked atleast few other classical music composers like Schubert, Mozart, Chopin, Bach, etc. Do you not think if you had used lift instead of just support, you might have arrived at a different answer? Afterall lift and leverage are more reliable measures than support and confidence.

2.3. 6/15
-What about lift?
-The graph doesn’t show a path from Beethoven to Eminem. The paths seem to end at either Eminem or Beethoven.
-You only pick out itemsets that contain Eminem or Beethoven. The confidence/lift between two artists will not be representative if you do this.
The rule Massive Attack -> Nelly Furtado in this case would have a different confidence/lift because of this and may be higher than it actually is.
-If you are going to filter by artist only generate rules starting from that artist e.g. by filtering by artist Beethoven the confidence value of rules of the form Beethoven -> {other artist} would still be applicable to the complete set of itemsets.

2.4. 6/15
-Interesting difference between the whole data set and filtered data set only including users of male teenagers.
-What age does a teenager have?
-Recommended to sample after discarding all the incomplete cases.
-No actual rules, you’ve generated most frequent items after filtering by an user group.


2.5 [7/15]
Alignment of columns in table needs to be fixed. It is difficult to follow otherwise. And how would you select the winner after putting these artists/bands in the questionnaire? Some weighted average or normal average? More explanation is needed here. 

Bonus 7/10

Good visualisations created and it is clear you understand the importance of them. The only thing missing would be to haved used the visualisation to identify something that wasn’t noticeable before (e.g. a new connection or an artist that connects varying genres).
