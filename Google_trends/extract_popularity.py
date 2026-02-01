"""
从Google Trends数据中提取每位选手的热度系数
热度系数 = 选手名列求和 / Dancing with the Stars列求和
"""

import pandas as pd
import os
from pathlib import Path

def calculate_popularity_coefficient(csv_path, celebrity_name):
    """
    计算单个选手的热度系数
    热度系数 = 选手名列的求和 / Dancing with the Stars列的求和
    
    Args:
        csv_path: CSV文件路径
        celebrity_name: 选手名字
    
    Returns:
        热度系数，如果文件不存在或计算失败则返回None
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 检查必要的列是否存在
        if celebrity_name not in df.columns:
            print(f"  警告: 列 '{celebrity_name}' 不在文件中")
            print(f"  文件列名: {list(df.columns)}")
            return None
        
        if 'Dancing with the Stars' not in df.columns:
            print(f"  警告: 列 'Dancing with the Stars' 不在文件中")
            return None
        
        # 计算求和
        celebrity_sum = df[celebrity_name].sum()
        dwts_sum = df['Dancing with the Stars'].sum()
        
        # 避免除以零
        if dwts_sum == 0:
            print(f"  警告: 'Dancing with the Stars' 列求和为0")
            return 0.0
        
        coefficient = celebrity_sum / dwts_sum
        return coefficient
    
    except FileNotFoundError:
        print(f"  文件不存在: {csv_path}")
        return None
    except Exception as e:
        print(f"  处理文件时出错: {e}")
        return None


def main():
    # 设置路径
    base_dir = Path(__file__).parent
    data_csv_path = base_dir / "2026_MCM_Problem_C_Data.csv"
    get_data_dir = base_dir / "get_data"
    output_path = base_dir / "popularity_coefficients.csv"
    
    # 读取主数据文件
    print(f"读取数据文件: {data_csv_path}")
    main_df = pd.read_csv(data_csv_path)
    
    # 提取选手名字和季数
    celebrities = main_df[['celebrity_name', 'season']].copy()
    print(f"共有 {len(celebrities)} 位选手")
    
    # 存储结果
    results = []
    
    # 遍历每位选手
    for idx, row in celebrities.iterrows():
        celebrity_name = row['celebrity_name']
        season = int(row['season'])
        
        # 构建CSV文件路径: get_data/season_XX/人名.csv
        # 处理文件名中的非法字符：引号替换为下划线，去掉末尾的点和空格
        safe_name = celebrity_name.replace('"', '_').rstrip('. ')
        season_folder = f"season_{season:02d}"
        csv_filename = f"{safe_name}.csv"
        csv_path = get_data_dir / season_folder / csv_filename
        
        print(f"处理: 第{season}季 - {celebrity_name}")
        
        # 计算热度系数
        coefficient = calculate_popularity_coefficient(csv_path, celebrity_name)
        
        results.append({
            'season': season,
            'celebrity_name': celebrity_name,
            'popularity_coefficient': coefficient
        })
        
        if coefficient is not None:
            print(f"  热度系数: {coefficient:.4f}")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 按季数和选手名排序
    result_df = result_df.sort_values(['season', 'celebrity_name'])
    
    # 保存结果
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_path}")
    
    # 打印统计信息
    successful = result_df['popularity_coefficient'].notna().sum()
    failed = result_df['popularity_coefficient'].isna().sum()
    print(f"\n统计:")
    print(f"  成功计算: {successful} 人")
    print(f"  计算失败: {failed} 人")
    
    # 按季显示结果摘要
    print("\n各季热度系数汇总:")
    for season in sorted(result_df['season'].unique()):
        season_data = result_df[result_df['season'] == season]
        valid_data = season_data[season_data['popularity_coefficient'].notna()]
        if len(valid_data) > 0:
            avg_coef = valid_data['popularity_coefficient'].mean()
            print(f"  第{season}季: {len(valid_data)}人, 平均热度系数: {avg_coef:.4f}")
        else:
            print(f"  第{season}季: 无有效数据")
    
    return result_df


if __name__ == "__main__":
    result = main()
