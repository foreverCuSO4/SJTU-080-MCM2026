import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt

def compare_trends():
    # 1. 初始化连接
    # hl='en-US' 设定语言为英语，tz=360 为美国中部时区
    pytrends = TrendReq(hl='en-US', tz=360)

    # 2. 定义查询参数
    # 关键词列表
    kw_list = ["Kate Flannery", "James Van Der Beek"]
    
    # 设定精确的时间范围：格式必须是 'YYYY-MM-DD YYYY-MM-DD'
    search_timeframe = '2019-09-16 2019-11-25'
    
    # 地区：美国
    search_geo = 'US'

    print(f"正在获取数据: {kw_list} 在 {search_geo} 地区，时间: {search_timeframe}...")

    # 3. 构建请求 (Build Payload)
    try:
        pytrends.build_payload(
            kw_list, 
            cat=0, 
            timeframe=search_timeframe, 
            geo=search_geo, 
            gprop=''
        )
        
        # 4. 获取随时间变化的兴趣度
        data = pytrends.interest_over_time()

        if data.empty:
            print("未找到数据，请检查网络连接或关键词拼写。")
            return

        # 5. 数据清洗
        # 删除 'isPartial' 列（该列用于标记数据是否不完整，通常不需要绘图）
        if 'isPartial' in data.columns:
            data = data.drop(columns=['isPartial'])

        print("\n--- 数据预览 (前5行) ---")
        print(data.head())
        
        # 打印统计摘要（平均热度对比）
        print("\n--- 热度统计摘要 ---")
        print(data.mean())

        # 6. 数据可视化
        plt.figure(figsize=(12, 6))
        
        # 绘制折线图
        for keyword in kw_list:
            plt.plot(data.index, data[keyword], label=keyword, marker='o', markersize=3)

        plt.title('Google Trends: Kate Flannery vs James Van Der Beek (US)')
        plt.xlabel('Date')
        plt.ylabel('Interest Score (0-100)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 自动调整日期标签格式
        plt.gcf().autofmt_xdate()
        
        # 显示图片
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")
        print("提示: 如果出现 429 错误，说明请求过于频繁，请稍后重试。")

if __name__ == "__main__":
    compare_trends()