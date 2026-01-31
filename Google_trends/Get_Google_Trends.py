"""
获取 Dancing With The Stars 明星的 Google Trends 数据
使用锚点归一化方法，使不同明星之间的数据可以比较
包含强大的反封锁机制和断点续传功能
"""

import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
import os
import time
import random
import json
import sys
from datetime import datetime
import urllib3

# 1. 禁用 SSL 警告（为了后面 verify=False 不报错）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. 强制设置系统级代理
# 请确保这里的端口号 (7890) 和你 VPN 软件里的一致！
# 如果是 v2rayN，可能是 10809
PROXY_PORT = '7890' 
os.environ['HTTP_PROXY'] = f'http://127.0.0.1:{PROXY_PORT}'
os.environ['HTTPS_PROXY'] = f'http://127.0.0.1:{PROXY_PORT}'

# ============== 配置参数 ==============
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEASON_TIME_FILE = os.path.join(DATA_DIR, 'season-time.csv')
CELEBRITY_DATA_FILE = os.path.join(DATA_DIR, '2026_MCM_Problem_C_Data.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'get_data')

# 锚点关键词 - 用于归一化数据
ANCHOR_KEYWORD = "Dancing with the Stars"

# 批量爬取设置
BATCH_SIZE = 4  # 每批最多4个明星（加上锚点共5个，是Google Trends的上限）

# 请求延迟设置（秒）
MIN_DELAY = 10  # 最小延迟
MAX_DELAY = 20  # 最大延迟

# 咖啡休息设置
COFFEE_BREAK_INTERVAL = 10  # 每处理10个明星休息一次
COFFEE_BREAK_DURATION = 60  # 咖啡休息时长（秒）

# 重试设置
MAX_RETRIES = 5  # 最大重试次数
RATE_LIMIT_SLEEP = 300  # 遇到429错误时的等待时间（5分钟）
NORMAL_RETRY_DELAY = 30  # 普通错误的重试延迟


def countdown_sleep(seconds, message="休息中"):
    """带倒计时显示的休眠函数"""
    for remaining in range(int(seconds), 0, -1):
        sys.stdout.write(f"\r  {message}... 剩余 {remaining} 秒   ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 50 + "\r")  # 清除倒计时行
    sys.stdout.flush()


def load_season_times():
    """从CSV加载赛季播出时间段"""
    df = pd.read_csv(SEASON_TIME_FILE)
    season_times = {}
    for _, row in df.iterrows():
        season = int(row['Season'])
        start_date = row['Start_Date']
        end_date = row['End_Date']
        season_times[season] = (start_date, end_date)
    print(f"✓ 已加载 {len(season_times)} 个赛季的时间数据")
    return season_times


def load_celebrities():
    """从CSV加载明星数据"""
    df = pd.read_csv(CELEBRITY_DATA_FILE)
    celebrities = df[['celebrity_name', 'season']].copy()
    celebrities['season'] = celebrities['season'].astype(int)
    print(f"✓ 已加载 {len(celebrities)} 条明星记录")
    return celebrities


def create_fresh_pytrends():
    """
    创建新的pytrends会话（每次请求重新初始化以避免封锁）
    
    注意：不使用 retries 和 backoff_factor 参数，因为在新版本 urllib3 中
    这些参数可能导致 'method_whitelist' 兼容性问题。
    我们在外层代码中手动处理重试逻辑。
    """

    return TrendReq(
        hl='en-US', 
        tz=360,
        timeout=(10, 25),  # 连接超时10秒，读取超时25秒
        retries=2,  # 简单重试次数
    )


def get_google_trends_with_anchor(celebrity_names, start_date, end_date, geo='US'):
    """
    使用锚点方法批量获取Google Trends数据
    
    Args:
        celebrity_names: 明星姓名列表（最多4个）
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        geo: 地区（默认：美国）
    
    Returns:
        包含归一化数据的DataFrame，失败返回None
        返回的DataFrame包含所有明星的数据和各自的归一化分数
    """
    if isinstance(celebrity_names, str):
        celebrity_names = [celebrity_names]
    
    if len(celebrity_names) > BATCH_SIZE:
        print(f"  ⚠ 警告: 一次最多查询{BATCH_SIZE}个明星，当前{len(celebrity_names)}个")
        celebrity_names = celebrity_names[:BATCH_SIZE]
    
    timeframe = f'{start_date} {end_date}'
    # 明星列表 + 锚点关键词（最多5个）
    kw_list = celebrity_names + [ANCHOR_KEYWORD]
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # 每次请求创建新的session
            pytrends = create_fresh_pytrends()
            
            pytrends.build_payload(
                kw_list,
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop=''
            )
            
            data = pytrends.interest_over_time()
            
            if data.empty:
                print(f"  ⚠ 未找到数据")
                return None
            
            # 移除 'isPartial' 列
            if 'isPartial' in data.columns:
                data = data.drop(columns=['isPartial'])
            
            # 为每个明星计算归一化分数
            for celeb_name in celebrity_names:
                if celeb_name in data.columns:
                    norm_col = f'{celeb_name}_normalized'
                    data[norm_col] = (data[celeb_name] / data[ANCHOR_KEYWORD]) * 100
                    # 处理除零产生的无穷值和NaN
                    data[norm_col] = data[norm_col].replace([float('inf'), float('-inf')], 0)
                    data[norm_col] = data[norm_col].fillna(0)
            
            return data
            
        except ResponseError as e:
            last_error = e
            error_msg = str(e)
            
            if '429' in error_msg:
                print(f"  ⛔ 遇到429错误（请求过于频繁）")
                if attempt < MAX_RETRIES - 1:
                    countdown_sleep(RATE_LIMIT_SLEEP, "等待API限制解除")
                    print(f"  🔄 重试中... (尝试 {attempt + 2}/{MAX_RETRIES})")
            else:
                print(f"  ❌ API错误: {error_msg}")
                if attempt < MAX_RETRIES - 1:
                    countdown_sleep(NORMAL_RETRY_DELAY, "等待重试")
                    print(f"  🔄 重试中... (尝试 {attempt + 2}/{MAX_RETRIES})")
                    
        except Exception as e:
            last_error = e
            print(f"  ❌ 未知错误: {e}")
            if attempt < MAX_RETRIES - 1:
                countdown_sleep(NORMAL_RETRY_DELAY, "等待重试")
                print(f"  🔄 重试中... (尝试 {attempt + 2}/{MAX_RETRIES})")
    
    print(f"  ✗ 所有重试均失败: {last_error}")
    return None


def save_batch_data(data, celebrity_names, season, output_dir):
    """
    从批量数据中提取并保存每个明星的数据
    
    Args:
        data: 包含所有明星数据的DataFrame
        celebrity_names: 明星姓名列表
        season: 赛季号
        output_dir: 输出目录
    
    Returns:
        saved_names: 成功保存的明星名字列表
    """
    saved_names = []
    
    for celeb_name in celebrity_names:
        if celeb_name not in data.columns:
            print(f"  ⚠ '{celeb_name}' 在返回数据中不存在，跳过")
            continue
        
        # 提取该明星的数据
        norm_col = f'{celeb_name}_normalized'
        cols_to_save = [celeb_name, ANCHOR_KEYWORD]
        if norm_col in data.columns:
            cols_to_save.append(norm_col)
        
        celeb_data = data[cols_to_save].copy()
        # 重命名归一化列为统一的 'normalized_score'
        if norm_col in celeb_data.columns:
            celeb_data = celeb_data.rename(columns={norm_col: 'normalized_score'})
        
        # 创建赛季子目录
        season_dir = os.path.join(output_dir, f'season_{season:02d}')
        os.makedirs(season_dir, exist_ok=True)
        
        # 保存文件
        filepath = get_celebrity_filepath(celeb_name, season, output_dir)
        celeb_data.to_csv(filepath)
        saved_names.append(celeb_name)
        
        # 打印统计摘要
        mean_raw = celeb_data[celeb_name].mean()
        max_raw = celeb_data[celeb_name].max()
        if 'normalized_score' in celeb_data.columns:
            mean_norm = celeb_data['normalized_score'].mean()
            max_norm = celeb_data['normalized_score'].max()
            print(f"    📈 {celeb_name}: 原始(均值={mean_raw:.1f}, 最大={max_raw}) | 归一化(均值={mean_norm:.2f}, 最大={max_norm:.2f})")
        else:
            print(f"    📈 {celeb_name}: 原始(均值={mean_raw:.1f}, 最大={max_raw})")
    
    return saved_names


def sanitize_filename(name):
    """将名称转换为安全的文件名"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip(' .')


def get_celebrity_filepath(celebrity_name, season, output_dir):
    """获取明星数据文件的完整路径"""
    season_dir = os.path.join(output_dir, f'season_{season:02d}')
    safe_name = sanitize_filename(celebrity_name)
    filename = f'{safe_name}.csv'
    return os.path.join(season_dir, filename)


def check_file_exists_and_valid(filepath, min_rows=2):
    """
    检查文件是否存在且内容有效
    
    Args:
        filepath: 文件路径
        min_rows: 最少行数（包括表头），默认2表示至少有1行数据
    
    Returns:
        bool: 文件存在且有效返回True，否则返回False
    """
    if not os.path.exists(filepath):
        return False
    
    try:
        # 检查文件大小（空文件或只有表头的文件通常很小）
        file_size = os.path.getsize(filepath)
        if file_size < 50:  # 小于50字节认为无效
            return False
        
        # 尝试读取并检查行数
        df = pd.read_csv(filepath)
        if len(df) < (min_rows - 1):  # -1 因为表头不算数据行
            return False
        
        return True
    except Exception:
        return False


def save_celebrity_data(data, celebrity_name, season, output_dir):
    """保存明星的trends数据到CSV文件"""
    # 创建赛季子目录
    season_dir = os.path.join(output_dir, f'season_{season:02d}')
    os.makedirs(season_dir, exist_ok=True)
    
    # 获取文件路径
    filepath = get_celebrity_filepath(celebrity_name, season, output_dir)
    
    # 保存到CSV
    data.to_csv(filepath)
    print(f"  💾 已保存: {filepath}")


def load_progress(output_dir):
    """加载进度文件以支持断点续传"""
    progress_file = os.path.join(output_dir, 'progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def save_progress(output_dir, progress):
    """保存进度文件"""
    progress_file = os.path.join(output_dir, 'progress.json')
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def get_celebrity_key(celebrity_name, season):
    """为明星-赛季组合创建唯一键"""
    return f"{celebrity_name}|{season}"


def random_delay():
    """随机延迟以避免被封锁"""
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    countdown_sleep(delay, "请求间隔")


def main():
    """主函数：协调数据收集流程（批量模式）"""
    print("=" * 65)
    print("  Dancing With The Stars - Google Trends 数据收集器")
    print("  使用锚点归一化方法 + 批量爬取模式（每批最多4个明星）")
    print("=" * 65)
    print()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    season_times = load_season_times()
    celebrities = load_celebrities()
    
    # 加载进度
    progress = load_progress(OUTPUT_DIR)
    completed_set = set(progress['completed'])
    
    print(f"\n📊 进度统计:")
    print(f"   已完成: {len(completed_set)} 位明星")
    print(f"   之前失败: {len(progress['failed'])} 位明星")
    print(f"   锚点关键词: \"{ANCHOR_KEYWORD}\"")
    print(f"   批量大小: {BATCH_SIZE} 明星/批")
    print()
    
    # 按赛季分组明星
    celebrities_by_season = {}
    for _, row in celebrities.iterrows():
        season = row['season']
        if season not in celebrities_by_season:
            celebrities_by_season[season] = []
        celebrities_by_season[season].append(row['celebrity_name'])
    
    # 统计
    total_celebrities = len(celebrities)
    success_count = 0
    fail_count = 0
    skip_count = 0
    batch_count = 0  # 实际爬取的批次数（用于咖啡休息计数）
    
    # 按赛季处理
    sorted_seasons = sorted(celebrities_by_season.keys())
    
    for season in sorted_seasons:
        celeb_list = celebrities_by_season[season]
        
        # 检查赛季是否存在
        if season not in season_times:
            print(f"\n⚠ 第 {season} 季不在 season-time.csv 中，跳过该赛季所有明星")
            for celeb_name in celeb_list:
                progress['failed'].append({
                    'name': celeb_name, 
                    'season': season, 
                    'reason': '赛季未找到'
                })
                fail_count += 1
            continue
        
        start_date, end_date = season_times[season]
        
        # 过滤出未完成的明星
        pending_celebs = []
        for celeb_name in celeb_list:
            key = get_celebrity_key(celeb_name, season)
            filepath = get_celebrity_filepath(celeb_name, season, OUTPUT_DIR)
            
            if key in completed_set:
                skip_count += 1
                continue
            
            if check_file_exists_and_valid(filepath):
                # 文件存在但不在completed_set中，同步更新
                progress['completed'].append(key)
                completed_set.add(key)
                save_progress(OUTPUT_DIR, progress)
                skip_count += 1
                continue
            
            pending_celebs.append(celeb_name)
        
        if not pending_celebs:
            print(f"\n第 {season} 季: 所有 {len(celeb_list)} 位明星已完成，跳过")
            continue
        
        print(f"\n{'='*50}")
        print(f"第 {season} 季 ({start_date} 至 {end_date})")
        print(f"待处理: {len(pending_celebs)}/{len(celeb_list)} 位明星")
        print(f"{'='*50}")
        
        # 将待处理明星分批（每批最多BATCH_SIZE个）
        batches = [pending_celebs[i:i+BATCH_SIZE] for i in range(0, len(pending_celebs), BATCH_SIZE)]
        
        for batch_idx, batch in enumerate(batches):
            batch_count += 1
            
            print(f"\n  📦 批次 {batch_idx + 1}/{len(batches)}: {batch}")
            
            # 获取数据
            data = get_google_trends_with_anchor(batch, start_date, end_date)
            
            if data is not None:
                # 保存每个明星的数据
                saved_names = save_batch_data(data, batch, season, OUTPUT_DIR)
                
                for celeb_name in batch:
                    key = get_celebrity_key(celeb_name, season)
                    if celeb_name in saved_names:
                        progress['completed'].append(key)
                        completed_set.add(key)
                        success_count += 1
                        print(f"    ✓ {celeb_name} 保存成功")
                    else:
                        progress['failed'].append({
                            'name': celeb_name, 
                            'season': season, 
                            'reason': '数据提取失败'
                        })
                        fail_count += 1
                        print(f"    ✗ {celeb_name} 保存失败")
            else:
                # 整批失败
                for celeb_name in batch:
                    key = get_celebrity_key(celeb_name, season)
                    progress['failed'].append({
                        'name': celeb_name, 
                        'season': season, 
                        'reason': '批次请求失败'
                    })
                    fail_count += 1
                print(f"    ✗ 整批请求失败")
            
            # 保存进度
            save_progress(OUTPUT_DIR, progress)
            
            # 检查是否需要咖啡休息（每处理一定数量的批次）
            if batch_count > 0 and batch_count % COFFEE_BREAK_INTERVAL == 0:
                print(f"\n  ☕ 咖啡休息时间！(已处理 {batch_count} 批)")
                countdown_sleep(COFFEE_BREAK_DURATION, "咖啡休息")
                print(f"  ✓ 休息完毕，继续工作...")
            
            # 每批爬取后随机延迟（防止被封）
            random_delay()
    
    # 最终总结
    print("\n" + "=" * 65)
    print("  数据收集完成！")
    print("=" * 65)
    print(f"  总明星数: {total_celebrities}")
    print(f"  ✓ 成功获取: {success_count}")
    print(f"  ⏭ 跳过（已完成）: {skip_count}")
    print(f"  ✗ 失败: {fail_count}")
    print(f"  📦 总批次数: {batch_count}")
    print(f"\n  数据保存位置: {OUTPUT_DIR}")
    
    # 保存最终进度
    save_progress(OUTPUT_DIR, progress)
    
    # 创建汇总文件
    create_summary(OUTPUT_DIR, season_times, celebrities, progress)


def create_summary(output_dir, season_times, celebrities, progress):
    """创建包含所有收集数据信息的汇总CSV"""
    summary_file = os.path.join(output_dir, 'collection_summary.csv')
    
    summary_data = []
    for _, row in celebrities.iterrows():
        celebrity_name = row['celebrity_name']
        season = row['season']
        key = get_celebrity_key(celebrity_name, season)
        
        status = 'completed' if key in progress['completed'] else 'failed'
        start_date, end_date = season_times.get(season, ('N/A', 'N/A'))
        
        summary_data.append({
            'celebrity_name': celebrity_name,
            'season': season,
            'start_date': start_date,
            'end_date': end_date,
            'status': status
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"\n  📋 汇总文件已保存: {summary_file}")


if __name__ == '__main__':
    main()
