import argparse
from data import covidData
from model_quantile import covidLSTMQuantiles, ensemble_results, \
                           organize_csv_for_forecast_hub
from dl_data import download_data
from datetime import date, timedelta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str,
                        help="current date, ie. include data up to which date")
    parser.add_argument("-l", "--lag", type=int, default=9,
                        help="temporal lag, ie. how many prior weeks' data")
    parser.add_argument("-p", "--print", type=bool, default=False,
                        help="print details of data and models")
    parser.add_argument("-e", "--epochs", type=int, default=15,
                        help="no epochs for training")
    parser.add_argument("-m", "--models", type=int, default=10,
                        help="no models in ensemble")
    return parser.parse_args()


def main():
    args = parse_args()

    # download data and save as csv
    download_data(args.date)

    for hz in range(1,5):
        # create data object from csv
        data = covidData(temporal_lag=args.lag, current_date=args.date,
                         forecast_horizon=hz)
        # preprocess
        data.load_dataframes(print_info=args.print)

        # create train-test split
        data_dictionary = data.create_train_test_datasets(print_info=args.print)

        X_train_ts = data_dictionary["X_train_ts"]
        y_train = data_dictionary["y_train"]
        X_test_ts = data_dictionary["X_test_ts"]
        test_info = data_dictionary["test_info"]

        n_temporal_fts = X_train_ts[0].shape[-1]
        if args.print:
            print("Number of temporal fts:", n_temporal_fts)

        n_train = X_train_ts.shape[0]
        n_test = X_test_ts.shape[0]
        if args.print:
            print("No training instances:", n_train,
                  ", No test instances:", n_test)


        assert X_test_ts.shape[0]==test_info.shape[0]

        for i in range(1, args.models+1):
            print("Training model "+str(i)+" of "+str(args.models))

            # initialise ensemble
            model = covidLSTMQuantiles(temporal_lag=args.lag,
                                       no_temporal_ft=n_temporal_fts)

            # train
            model.train(X_ts=X_train_ts, y=y_train, no_epochs=args.epochs)

            # test
            results = model.predict(X_ts=X_test_ts, test_info_df=test_info)


            forecast_date = date(int(args.date.split("-")[0]),
                                 int(args.date.split("-")[1]),
                                 int(args.date.split("-")[2]))
            target_date = forecast_date + timedelta(days=6)
            target_date_str = target_date.strftime("%Y-%m-%d")
            if args.print:
                print("Target date:", target_date_str)

            model.save_results(current_date=args.date,
                               target_date=target_date_str,
                               test_horizon=hz,
                               print_info=args.print,
                               run=i)

        ensemble_results(current_date=args.date,
                         test_horizon=hz,
                         lag=args.lag,
                         ensemble_sz=args.models)

        organize_csv_for_forecast_hub(args.date, hz)


if __name__ == '__main__':
    main()
