import pandas as pd
import datetime
import yfinance as yf
import os
import boto3
from io import StringIO
# --- ヘルパー関数 ---

def calculate_streak(series):
    """条件(True/False)が連続する回数を計算する。"""
    sign = (series > 0).astype(int).replace(0, -1)
    group = (sign != sign.shift()).cumsum()
    counts = sign.groupby([group, sign]).cumcount() + 1
    return sign * counts

def get_scalar(series_val):
    """Seriesからスカラー値に安全に変換する。NaNの場合は0を返す。"""
    if pd.isna(series_val).any():
        return 0
    return series_val.item()

# --- データ処理関数 ---

def load_tickers(file_path):
    """銘柄リストのCSVファイルを読み込む。"""
    try:
        df_tickers = pd.read_csv(file_path)
        print(f"'{file_path}' から {len(df_tickers)} 銘柄を読み込みました。")
        return df_tickers
    except FileNotFoundError:
        print(f"エラー: 銘柄リストファイル '{file_path}' が見つかりません。")
        return None

def analyze_single_stock(company_code, company_name, start_day, end_day):
    """1つの銘柄を分析し、結果をリストで返す。"""
    print(f"分析中: {company_code} {company_name}")
    df = yf.download(f"{company_code}.T", start=start_day, end=end_day, progress=False)
    if df.empty:
        print(f"警告: {company_code} のデータを取得できませんでした。スキップします。")
        return None

    try:
        price = df['Close'].iloc[-1].item()
        volume_0 = df["Volume"].iloc[-1].item()
        volume_1 = df["Volume"].iloc[-2].item()
        volume_2 = df["Volume"].iloc[-3].item()
    except IndexError:
        print(f"警告: {company_code} のデータが不十分です。スキップします。")
        return None

    # 移動平均線（月足）
    month = df[['Close']].resample('ME').last()
    month["Month_20"] = month.rolling(window=20, min_periods=1).mean()
    month["Month_First"] = df[['Open']].resample('ME').first()
    month["Month_Last"] = df[['Close']].resample('ME').last()

    # 連続月数の計算 (20ヶ月線)
    month['streak_20m'] = calculate_streak(month['Close'].squeeze() - month['Month_20'].squeeze())

    # 移動平均線（日足）
    df['sma07'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['sma20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['sma60'] = df['Close'].rolling(window=60, min_periods=1).mean()

    # 連続日数の計算 (7日線, 20日線)
    df['streak_20d'] = calculate_streak(df['Close'].squeeze() - df['sma20'].squeeze())
    df['streak_7d'] = calculate_streak(df['Close'].squeeze() - df['sma07'].squeeze())

    # 出来高前日比の計算
    volume_ratio = 0
    if volume_1 > 0: # ゼロ除算を避ける
        volume_ratio = int(volume_0 / volume_1 * 100)

    # 最新のデータを取得してリストに追加
    latest_month = month.iloc[-1]
    latest_day = df.iloc[-1]

    return [
        company_code, company_name, int(price),
        1 if get_scalar(latest_month['streak_20m']) > 0 else -1, get_scalar(latest_month['streak_20m']),
        1 if get_scalar(latest_day['streak_20d']) > 0 else -1, get_scalar(latest_day['streak_20d']),
        1 if get_scalar(latest_day['streak_7d']) > 0 else -1, get_scalar(latest_day['streak_7d']),
        int(volume_0), int(volume_1), int(volume_2), # 確実にint型に変換
        volume_ratio, datetime.datetime.today().date()
    ]

def save_to_s3(df, bucket_name, object_key):
    """分析結果のDataFrameをS3にCSVとしてアップロードする。"""
    try:
        # GitHub ActionsのSecretsから認証情報を取得
        aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
        aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')

        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())
        print(f"分析結果をS3バケット '{bucket_name}' の '{object_key}' にアップロードしました。")
    except Exception as e:
        print(f"\nS3へのアップロード中にエラーが発生しました: {e}")

def main():
    """
    メイン処理：銘柄リストの読み込み、分析、データベースへの保存を実行する。
    """
    print("株価データの分析とデータベースへの保存を開始します...")

    # --- 1. 初期設定 ---
    TICKER_FILE = "meigara_400.csv"
    S3_BUCKET = 'swing-trade-data'
    S3_KEY = 'jpx400.csv'
    # データ取得期間の設定
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    start_day = datetime.datetime.now(JST) - datetime.timedelta(days=1100)
    end_day = datetime.datetime.now(JST)

    # --- 2. 銘柄リストの読み込み ---
    df_tickers = load_tickers(TICKER_FILE)
    if df_tickers is None:
        return

    # --- 3. 各銘柄のテクニカル分析（メインループ） ---
    all_results = []
    for index, row in df_tickers.iterrows():
        company_code = str(row['銘柄コード'])
        company_name = str(row['銘柄名'])
        
        result = analyze_single_stock(company_code, company_name, start_day, end_day)
        if result:
            all_results.append(result)

    # --- 4. DataFrameの作成とS3へのアップロード ---
    if not all_results:
        print("分析対象のデータがありませんでした。")
        return

    columns = [
        '銘柄コード', '銘柄名', '価格', '月足20_flag', '月20数', '日足20_flag', '日20数',
        '日足7_flag', '日7数', '出来高_0', '出来高_1', '出来高_2', '出来高_前日比', '取得日'
    ]
    df_results = pd.DataFrame(all_results, columns=columns)
    save_to_s3(df_results, S3_BUCKET, S3_KEY)

if __name__ == "__main__":
    main()