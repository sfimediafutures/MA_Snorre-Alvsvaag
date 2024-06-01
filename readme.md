## MA_Snorre-Alvsvaag
The code in this repository is connected to my Master thesis: Addressing the Next-Poster Problem: A Hybrid Recommender System for Streaming Platforms

Abstract:
Recommendation strategies in the Movie domain is varied, but has been shown to aid
users in finding content they like. On video streaming-platforms such as TV 2 Play, the
user is exposed to several different arenas where they can find something they would
like to watch, be that the front-page of the site, or a suggestion for something more to
watch after they concluded a movie or series. These are all areas where Recommender
strategies can recommend something based on either the preferences of the user, or in
the case of the concluded movie or series, something more to watch based on what they
just watched. This latter aspects, being what I refer to as the next-poster problem in this
thesis, is not a largely explored area of research, where previous actors have simply utilized
the already established Collaborative Filtering (CF) model concerned with the user’s
preferences without considering what the user just watched. Here I show that a solution to
the next-poster problem is to combine the CF model with a Sequence Aware approach based
on Markov Chains, finding an increase in implied user satisfaction over the baseline CF
approach. Through an online evaluation on the streaming platform TV 2 Play, I show that
using a Hybrid approach to solve the next-poster problem rather than a traditional CF model
leads to a lessening in user engagement such as CTR, but an increase in the clicks resulting
in a user actually watching the content, this being our implied user satisfaction. Further
as a result of this online evaluation, I am able to show that its possible to find the best
configuration for a Hybrid model based on Sequence Aware and CF approaches deployed
in a real life scenario, through offline evaluation. The results allows me to showcase the
importance of considering Sequence of items when recommending for the next-poster
problem, and to show that an offline evaluation can imply results in a real world scenario,
when considering the Movie domain. Although an improvement, this thesis also shows that
there are many more avenues to consider for the next-poster problem.

---
USAGE:
In this repository you will find the `rec` python package created to evaluate a Collaborative Filtering model, Markov Model, and a Hybrid of the two. 
It contains tools to calculate popularity in datasets, and evaluate MRR, CTR, Coverage and Popularity measures. See main.py for how it can be used.

There is also a tool to allow you to get notified through Slack, add the env variables:
```
SLACK_URL=...
SLACK_CHANNEL=...
```

You dont need a Slack bot for this to work, as we are only sending messages, be warned the code is messy, but gets the job done. An evaluatin of the results, as well as the actual results of the offline evaluation conducted in this thesis can be found in the `results` folder. 

---

This work was supported by industry partners and the Research Council of Norway
with funding to MediaFutures: Research Centre for Responsible Media Technology and
Innovation, through the Centres for Research-based Innovation scheme, project number
309339.

