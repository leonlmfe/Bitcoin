from optparse import OptionParser
from game import Game, HumanModel, FileLogger
from ai_bfs_model import AIBFSModel
from ai_model import AIModel

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--model", action="store", type="string", dest="model", default="human", help="which model in use [ai, ai_bfs, human]")
    parser.add_option("--log-file", action="store", type="string", dest="log_file")
    parser.add_option("--fps", action="store", type="int", dest="fps", default=10)
    parser.add_option("--play_turns", action="store", type="int", dest="play_turns")

    (options, args) = parser.parse_args()

    if args:
        parser.error("Invalid arguments in execution")
    else:
        model = None
        logger = None

        if options.log_file:
            logger = FileLogger()
        if options.model == "human":
            model = HumanModel(logger)
        elif options.model == "ai_bfs":
            model = AIBFSModel(logger)
        elif options.model == "ai":
            model = AIModel(logger)
        else:
            parser.error("Invalid model is in use")

       
        game = Game()
        game.loop_with_model(model, options.fps, options.play_turns)
       
        if logger:
            logger.save(options.log_file)
