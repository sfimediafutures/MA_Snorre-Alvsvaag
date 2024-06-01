from rec.models.als import CFRecommender
from rec.models.bridges import Bridges
from rec.models.reranker import Reranker
import logging
import colorlog
from rec.evaluator.evaluator import Evaluation
from rec.utils.popularity import PopularityScore
import threadpoolctl
import os
from rec.utils.slack import Slack
import traceback
import sys

def beep(n=1, type='Blow'):
    for i in range(n):
        os.system(f'afplay /System/Library/Sounds/{type}.aiff')


# Initialize the models

if __name__ == '__main__':
    threadpoolctl.threadpool_limits(12, "blas")
    logger = colorlog.getLogger()
    logger.setLevel(logging.DEBUG)  
    
    slack = Slack()
    
    try:
        slack.send_message("Starting the evaluation script")
        # Create a StreamHandler to output logs to the terminal
        stream_handler = colorlog.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        # Create a formatter with colors
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'white',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        # Add the formatter to the handler
        stream_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(stream_handler)
        logger.info("Calculating popularity scores...")
        P = PopularityScore(logger=logger)
        P.load_data('./data/cf/train', nested=True, limit=1, type='viewing')
        P.calculate_popularity_scores(1000) # 1000 to consider all data

        PS= PopularityScore()
        PS.load_data('./data/bridges/train', nested=True, limit=1, type='sessions')
        PS.calculate_popularity_scores_sessions()

        logger.info("Fitting CF model...")
        CFR = CFRecommender(factors=1, use_gpu=False, use_cg=False, iterations=1, logger=logger)
        CFR.load_data('./data/cf/train', nested=True, limit=1)
        CFR.preprocess()
        CFR.fit()

        logger.info("Fitting Bridges model...")
        B = Bridges(method='frequencyScoreNormalizedLog2', logger=logger)
        B.fit(path='./data/bridges/train', nested=True, limit=1)

        logger.info("Fitting Reranker model...")
        R = Reranker(B, CFR, logger=logger)

        # beep(1, 'Blow') # I NEED TO BE REMOVED IF YOU WANNA RUN ME :)
        ## FIRST:
        # slack.send_message("Models are trained, starting the evaluation...") # I ALSO NEED TO BE REMOVED, UNLESS YOU ARE ON A MAC AND WANT A SLACK NOTIFICATION WHEN THE SCRIPT IS DONE :)
        experiment_id = 'final_full'
        out_path = './data/evaluations/'
        E = Evaluation(sample=True, sample_size=10000, out_path=out_path, logger=logger, popularity_scores=P.popularity_scores, session_popularity_scores=PS.popularity_scores, slack=slack)
        R = Reranker(B, CFR, logger=logger)
        E.setup(CFR, B, R, path='./data/testdata/test_dataset_filtered_cf_bridges.csv')
        # E.prepare_reranker_evaluations(["bridges"],['frequencyScoreNormalizedLog2'], [0.1], [20], [3, 10])
        E.prepare_reranker_evaluations(["reranker", "bridges", "cf"],['frequencyScore','frequencyScoreNormalizedLog2'], [0.1, 0.3, 0.5, 0.7, 0.9], [20, 50, 100], [1, 3, 5, 10, 20])
        E.evaluate_reranker(experiment_id)

        ## THEN (for days parameter):
        logger.info("Fitting Bridges model...")
        B = Bridges(method='frequencyScoreNormalizedLog2', logger=logger)
        B.fit(path='./data/bridges/train-short', nested=True, limit=-1)

        logger.info("Fitting Reranker model...")
        R = Reranker(B, CFR, logger=logger)
        E = Evaluation(sample=True, sample_size=1000000, out_path=out_path, logger=logger, popularity_scores=P.popularity_scores, session_popularity_scores=PS.popularity_scores, slack=slack)
        E.setup(CFR, B, R, path='./data/testdata/test_dataset_filtered_cf_bridges.csv')
        E.prepare_reranker_evaluations(["reranker", "bridges"],['frequencyScore','frequencyScoreNormalizedLog2'], [0.1, 0.3, 0.5, 0.7, 0.9], [20, 50, 100], [1, 3, 5, 10, 20])    
        E.evaluate_reranker(experiment_id + "_short")
        
        # beep(5, 'Blow') # I ALSO NEED TO BE REMOVED, UNLESS YOU ARE ON A MAC AND WANT A AUDIO NOTIFICATION WHEN THE SCRIPT IS DONE :)
    except Exception:
        slack.send_exception(sys.exc_info())
        logger.error("An exception occurred", exc_info=True)
        sys.exit(1)  # Close the app


