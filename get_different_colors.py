def generate_distinct_colors():
    colors = []
    step = 256 // 4  # 计算颜色通道的步长
    results = []
    # 生成16个差别明显的颜色
    for r in range(step, 256, step):
        for g in range(step, 256, step):
            for b in range(step, 256, step):
                color = (r, g, b)
                colors.append(color)
    for i in range(1,17):
        idx = i * 8 % 27
        results.append(colors[idx])
    return results  
colors = np.array(generate_distinct_colors()).reshape(16,3)
