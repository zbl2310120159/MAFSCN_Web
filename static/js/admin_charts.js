/**
 * MAFSCN 数据分析看板 - ECharts图表逻辑
 * 包含: 图表初始化、数据加载、渲染函数、景区选择器状态管理
 */

/* ============================================================
   全局状态
   ============================================================ */
var currentScenic = '全部';       // 当前选中景区
var currentSentiment = '正面';    // 当前关键词情感筛选
var charts = {};                   // ECharts实例集合
var scenicData = {};               // 景区数据缓存

/* 统一主题色 */
var COLORS = {
  primary: '#3b82f6',       // 主色蓝
  primaryLight: '#60a5fa',
  primaryDark: '#2563eb',
  positive: '#22c55e',      // 正面绿
  positiveLight: '#86efac',
  negative: '#ef4444',      // 负面红
  negativeLight: '#fca5a5',
  neutral: '#9ca3af',       // 中性灰
  neutralLight: '#d1d5db',
  text: '#1e293b',
  textLight: '#64748b',
  textMuted: '#94a3b8',
  border: '#e2e8f0',
  bg: '#f8fafc'
};

/* ============================================================
   初始化所有ECharts实例
   ============================================================ */
function initAllCharts() {
  // 全局图表
  charts.stats = echarts.init(document.getElementById('chart-stats'));
  charts.scoreDist = echarts.init(document.getElementById('chart-scoreDist'));
  charts.sentimentRatio = echarts.init(document.getElementById('chart-sentimentRatio'));
  charts.timeTrend = echarts.init(document.getElementById('chart-timeTrend'));
  charts.commentLength = echarts.init(document.getElementById('chart-commentLength'));
  charts.comparison = echarts.init(document.getElementById('chart-comparison'));

  // 景区图表
  charts.keywords = echarts.init(document.getElementById('chart-keywords'));
  charts.dimensions = echarts.init(document.getElementById('chart-dimensions'));
  charts.wordcloudPositive = echarts.init(document.getElementById('chart-wordcloud-positive'));
  charts.wordcloudNegative = echarts.init(document.getElementById('chart-wordcloud-negative'));

  // 窗口resize自适应
  window.addEventListener('resize', function() {
    Object.keys(charts).forEach(function(key) {
      if (charts[key]) charts[key].resize();
    });
  });
}

/* ============================================================
   加载状态管理
   ============================================================ */
function showLoading(id) {
  var el = document.getElementById('loading-' + id);
  if (el) el.classList.remove('hidden');
  var errEl = document.getElementById('error-' + id);
  if (errEl) errEl.classList.add('hidden');
  var emptyEl = document.getElementById('empty-' + id);
  if (emptyEl) emptyEl.classList.add('hidden');
}

function hideLoading(id) {
  var el = document.getElementById('loading-' + id);
  if (el) el.classList.add('hidden');
}

function showError(id) {
  hideLoading(id);
  var errEl = document.getElementById('error-' + id);
  if (errEl) errEl.classList.remove('hidden');
}

function showEmpty(id) {
  hideLoading(id);
  var emptyEl = document.getElementById('empty-' + id);
  if (emptyEl) emptyEl.classList.remove('hidden');
}

function hideEmpty(id) {
  var emptyEl = document.getElementById('empty-' + id);
  if (emptyEl) emptyEl.classList.add('hidden');
}

/* ============================================================
   重试加载
   ============================================================ */
function retryLoad(chartId) {
  // 全局图表重试
  if (['stats', 'scoreDist', 'sentimentRatio', 'timeTrend', 'commentLength', 'comparison'].indexOf(chartId) >= 0) {
    loadGlobalData();
  } else if (currentScenic !== '全部') {
    loadScenicData(currentScenic);
  }
}

/* ============================================================
   API请求封装
   ============================================================ */
async function apiGet(url) {
  var resp = await fetch(url);
  if (!resp.ok) throw new Error('HTTP ' + resp.status);
  var json = await resp.json();
  if (!json.success) throw new Error(json.error || '请求失败');
  return json.data;
}

/* ============================================================
   加载全局数据
   ============================================================ */
async function loadGlobalData() {
  // 并行请求所有全局数据
  var promises = [
    loadWithRetry('stats', function() { return apiGet('/api/admin/stats'); }),
    loadWithRetry('scoreDist', function() { return apiGet('/api/admin/score_dist'); }),
    loadWithRetry('sentimentRatio', function() { return apiGet('/api/admin/sentiment_ratio'); }),
    loadWithRetry('timeTrend', function() { return apiGet('/api/admin/time_trend'); }),
    loadWithRetry('commentLength', function() { return apiGet('/api/admin/comment_length'); }),
    loadWithRetry('comparison', function() { return apiGet('/api/admin/comparison'); })
  ];

  // 分别处理结果
  promises[0].then(function(data) { renderStatsChart(data); }).catch(function() { showError('stats'); });
  promises[1].then(function(data) { renderScoreDistChart(data); }).catch(function() { showError('scoreDist'); });
  promises[2].then(function(data) { renderSentimentRatioChart(data); }).catch(function() { showError('sentimentRatio'); });
  promises[3].then(function(data) { renderTimeTrendChart(data); }).catch(function() { showError('timeTrend'); });
  promises[4].then(function(data) { renderCommentLengthChart(data); }).catch(function() { showError('commentLength'); });
  promises[5].then(function(data) { renderComparisonChart(data); }).catch(function() { showError('comparison'); });
}

/* 带加载状态的请求封装 */
async function loadWithRetry(chartId, fetchFn) {
  showLoading(chartId);
  try {
    var data = await fetchFn();
    hideLoading(chartId);
    return data;
  } catch (err) {
    hideLoading(chartId);
    throw err;
  }
}

/* ============================================================
   加载景区数据
   ============================================================ */
async function loadScenicData(scenicName) {
  // 更新景区提示标签
  updateScenicHints(scenicName);

  // 隐藏空状态
  hideEmpty('keywords');
  hideEmpty('dimensions');
  hideEmpty('wordcloud');
  hideEmpty('usefulComments');

  // 并行请求景区数据
  var keywordsPromise = loadScenicApi('keywords', '/api/admin/keywords?scenic=' + encodeURIComponent(scenicName) + '&sentiment=' + encodeURIComponent(currentSentiment));
  var dimensionsPromise = loadScenicApi('dimensions', '/api/admin/dimensions?scenic=' + encodeURIComponent(scenicName));
  var wordcloudPromise = loadScenicApi('wordcloud', '/api/admin/wordcloud?scenic=' + encodeURIComponent(scenicName));
  var usefulPromise = loadScenicApi('usefulComments', '/api/admin/useful_comments?scenic=' + encodeURIComponent(scenicName) + '&top_k=5');

  keywordsPromise.then(function(data) { renderKeywordsChart(data); }).catch(function() { showError('keywords'); });
  dimensionsPromise.then(function(data) { renderDimensionsChart(data); }).catch(function() { showError('dimensions'); });
  wordcloudPromise.then(function(data) { renderWordcloudChart(data); }).catch(function() { showError('wordcloud'); });
  usefulPromise.then(function(data) { renderUsefulComments(data); }).catch(function() { showError('usefulComments'); });
}

/* 景区API请求封装 */
async function loadScenicApi(chartId, url) {
  showLoading(chartId);
  try {
    var data = await apiGet(url);
    hideLoading(chartId);
    return data;
  } catch (err) {
    hideLoading(chartId);
    throw err;
  }
}

/* ============================================================
   景区选择器
   ============================================================ */
function selectScenic(name, btnEl) {
  // 更新按钮状态
  var btns = document.querySelectorAll('.scenic-btn');
  btns.forEach(function(b) { b.classList.remove('active'); });
  btnEl.classList.add('active');

  currentScenic = name;

  if (name === '全部') {
    // 隐藏景区图表，显示空状态
    showEmpty('keywords');
    showEmpty('dimensions');
    showEmpty('wordcloud');
    showEmpty('usefulComments');
    // 清空景区图表
    if (charts.keywords) charts.keywords.clear();
    if (charts.dimensions) charts.dimensions.clear();
    if (charts.wordcloudPositive) charts.wordcloudPositive.clear();
    if (charts.wordcloudNegative) charts.wordcloudNegative.clear();
    document.getElementById('usefulCommentsList').innerHTML = '';
    updateScenicHints('全部');
  } else {
    loadScenicData(name);
  }
}

/* 更新景区提示标签 */
function updateScenicHints(scenicName) {
  var hintText = scenicName === '全部' ? '全部景区' : scenicName;
  var hintIds = ['keywordsScenicHint', 'dimensionsScenicHint', 'wordcloudScenicHint', 'usefulScenicHint'];
  hintIds.forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.textContent = hintText;
  });
}

/* ============================================================
   情感切换按钮
   ============================================================ */
function switchSentiment(sentiment, btnEl) {
  // 更新按钮状态
  var btns = document.querySelectorAll('.sentiment-toggle-btn');
  btns.forEach(function(b) {
    b.className = 'sentiment-toggle-btn';
  });
  var activeClass = 'active-' + (sentiment === '正面' ? 'positive' : sentiment === '负面' ? 'negative' : 'neutral');
  btnEl.classList.add(activeClass);

  currentSentiment = sentiment;

  // 如果已选择景区，重新加载关键词
  if (currentScenic !== '全部') {
    showLoading('keywords');
    hideEmpty('keywords');
    apiGet('/api/admin/keywords?scenic=' + encodeURIComponent(currentScenic) + '&sentiment=' + encodeURIComponent(sentiment))
      .then(function(data) {
        hideLoading('keywords');
        renderKeywordsChart(data);
      })
      .catch(function() {
        showError('keywords');
      });
  }
}

/* ============================================================
   1. 各景区评论数量 - 柱状图
   ============================================================ */
function renderStatsChart(data) {
  if (!charts.stats || !data) return;
  // API返回: [{name, negative, neutral, positive, total}, ...]
  var items = Array.isArray(data) ? data : [];
  var labels = items.map(function(d) { return d.name; });
  var values = items.map(function(d) { return d.total; });

  charts.stats.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 }
    },
    grid: { left: 80, right: 20, top: 20, bottom: 60 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: {
        rotate: 35,
        fontSize: 11,
        color: COLORS.textLight
      },
      axisLine: { lineStyle: { color: COLORS.border } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 11, color: COLORS.textMuted },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    series: [{
      type: 'bar',
      data: values,
      barWidth: '50%',
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: COLORS.primary },
          { offset: 1, color: COLORS.primaryLight }
        ]),
        borderRadius: [4, 4, 0, 0]
      },
      emphasis: {
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: COLORS.primaryDark },
            { offset: 1, color: COLORS.primary }
          ])
        }
      }
    }],
    animationDuration: 800,
    animationEasing: 'cubicOut'
  });
}

/* ============================================================
   2. 评分分布 - 柱状图
   ============================================================ */
function renderScoreDistChart(data) {
  if (!charts.scoreDist || !data) return;
  // API返回: {by_scenic:[...], overall:[{score, count},...]}
  var overall = data.overall || [];
  var labels = overall.map(function(d) { return String(d.score); });
  var values = overall.map(function(d) { return d.count; });

  // 根据评分值设置颜色
  var colorMap = {
    '1': COLORS.negative,
    '2': '#f97316',
    '3': COLORS.neutral,
    '4': '#84cc16',
    '5': COLORS.positive
  };

  var itemColors = labels.map(function(label) {
    return colorMap[label] || COLORS.primary;
  });

  charts.scoreDist.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 }
    },
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      data: labels.map(function(l) { return l + '分'; }),
      axisLabel: { fontSize: 12, color: COLORS.textLight },
      axisLine: { lineStyle: { color: COLORS.border } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 11, color: COLORS.textMuted },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    series: [{
      type: 'bar',
      data: values.map(function(v, i) {
        return {
          value: v,
          itemStyle: { color: itemColors[i], borderRadius: [4, 4, 0, 0] }
        };
      }),
      barWidth: '50%'
    }],
    animationDuration: 800,
    animationEasing: 'cubicOut'
  });
}

/* ============================================================
   3. 情感占比分析 - 堆叠柱状图
   ============================================================ */
function renderSentimentRatioChart(data) {
  if (!charts.sentimentRatio || !data) return;
  // API返回: [{name, positive_pct, neutral_pct, negative_pct, ...}, ...]
  var items = Array.isArray(data) ? data : [];
  var scenics = items.map(function(d) { return d.name; });
  var positive = items.map(function(d) { return d.positive_pct; });
  var neutral = items.map(function(d) { return d.neutral_pct; });
  var negative = items.map(function(d) { return d.negative_pct; });

  charts.sentimentRatio.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 },
      formatter: function(params) {
        var tip = '<b>' + params[0].axisValue + '</b><br/>';
        params.forEach(function(p) {
          tip += '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + p.color + ';margin-right:6px;"></span>';
          tip += p.seriesName + ': <b>' + p.value.toFixed(1) + '%</b><br/>';
        });
        return tip;
      }
    },
    legend: {
      data: ['正面', '中性', '负面'],
      top: 0,
      textStyle: { fontSize: 12, color: COLORS.textLight }
    },
    grid: { left: 50, right: 20, top: 40, bottom: 60 },
    xAxis: {
      type: 'category',
      data: scenics,
      axisLabel: { rotate: 30, fontSize: 11, color: COLORS.textLight },
      axisLine: { lineStyle: { color: COLORS.border } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      max: 100,
      axisLabel: { fontSize: 11, color: COLORS.textMuted, formatter: '{value}%' },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    series: [
      {
        name: '正面',
        type: 'bar',
        stack: 'total',
        data: positive,
        itemStyle: { color: COLORS.positive, borderRadius: [0, 0, 0, 0] },
        barWidth: '45%'
      },
      {
        name: '中性',
        type: 'bar',
        stack: 'total',
        data: neutral,
        itemStyle: { color: COLORS.neutral }
      },
      {
        name: '负面',
        type: 'bar',
        stack: 'total',
        data: negative,
        itemStyle: { color: COLORS.negative, borderRadius: [4, 4, 0, 0] }
      }
    ],
    animationDuration: 800,
    animationEasing: 'cubicOut'
  });
}

/* ============================================================
   4. 评论时间趋势 - 折线图
   ============================================================ */
function renderTimeTrendChart(data) {
  if (!charts.timeTrend || !data) return;
  // API返回: {overall: [{month, negative, neutral, positive, total}, ...], by_scenic: [...]}
  var items = (data.overall || []).slice(-24); // 最近24个月
  var months = items.map(function(d) { return d.month; });
  var positive = items.map(function(d) { return d.positive; });
  var neutral = items.map(function(d) { return d.neutral; });
  var negative = items.map(function(d) { return d.negative; });

  charts.timeTrend.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 }
    },
    legend: {
      data: ['正面', '中性', '负面'],
      top: 0,
      textStyle: { fontSize: 12, color: COLORS.textLight }
    },
    grid: { left: 50, right: 20, top: 40, bottom: 40 },
    xAxis: {
      type: 'category',
      data: months,
      axisLabel: { fontSize: 11, color: COLORS.textLight, rotate: 30 },
      axisLine: { lineStyle: { color: COLORS.border } },
      axisTick: { show: false },
      boundaryGap: false
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 11, color: COLORS.textMuted },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    series: [
      {
        name: '正面',
        type: 'line',
        data: positive,
        smooth: true,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { color: COLORS.positive, width: 2 },
        itemStyle: { color: COLORS.positive }
      },
      {
        name: '中性',
        type: 'line',
        data: neutral,
        smooth: true,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { color: COLORS.neutral, width: 2 },
        itemStyle: { color: COLORS.neutral }
      },
      {
        name: '负面',
        type: 'line',
        data: negative,
        smooth: true,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { color: COLORS.negative, width: 2 },
        itemStyle: { color: COLORS.negative }
      }
    ],
    animationDuration: 1000,
    animationEasing: 'cubicOut'
  });
}

/* ============================================================
   5. 关键词TOP10 - 横向柱状图
   ============================================================ */
function renderKeywordsChart(data) {
  if (!charts.keywords || !data) return;
  // API返回: [{word, count}, ...]
  var items = Array.isArray(data) ? data : [];
  var keywords = items.map(function(d) { return d.word; });
  var counts = items.map(function(d) { return d.count; });

  // 反转使最大值在顶部
  var kwReversed = keywords.slice().reverse();
  var ctReversed = counts.slice().reverse();

  // 根据当前情感选择颜色
  var barColor = currentSentiment === '正面' ? COLORS.positive :
                 currentSentiment === '负面' ? COLORS.negative : COLORS.neutral;
  var barColorLight = currentSentiment === '正面' ? COLORS.positiveLight :
                      currentSentiment === '负面' ? COLORS.negativeLight : COLORS.neutralLight;

  charts.keywords.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 }
    },
    grid: { left: 90, right: 30, top: 10, bottom: 20 },
    xAxis: {
      type: 'value',
      axisLabel: { fontSize: 11, color: COLORS.textMuted },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'category',
      data: kwReversed,
      axisLabel: { fontSize: 12, color: COLORS.textLight },
      axisLine: { lineStyle: { color: COLORS.border } },
      axisTick: { show: false }
    },
    series: [{
      type: 'bar',
      data: ctReversed,
      barWidth: '55%',
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
          { offset: 0, color: barColorLight },
          { offset: 1, color: barColor }
        ]),
        borderRadius: [0, 4, 4, 0]
      }
    }],
    animationDuration: 600,
    animationEasing: 'cubicOut'
  });
}

/* ============================================================
   6. 多维评分雷达图
   ============================================================ */
function renderDimensionsChart(data) {
  if (!charts.dimensions || !data) return;
  // API返回: {name, dimensions: {风景:0.08, 服务:0.49, ...}}
  var dimObj = data.dimensions || {};
  var dimensions = Object.keys(dimObj);
  var scores = dimensions.map(function(d) { return Math.round((dimObj[d] || 0) * 10 * 10) / 10; }); // 转为0-10分制

  charts.dimensions.setOption({
    tooltip: {
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 }
    },
    radar: {
      indicator: dimensions.map(function(d) {
        return { name: d, max: 10 };
      }),
      shape: 'polygon',
      splitNumber: 5,
      axisName: {
        color: COLORS.textLight,
        fontSize: 12,
        fontWeight: 500
      },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
      splitArea: {
        areaStyle: { color: ['#fafbff', '#f1f5f9', '#e8eef6', '#f1f5f9', '#fafbff'] }
      },
      axisLine: { lineStyle: { color: '#e2e8f0' } }
    },
    series: [{
      type: 'radar',
      data: [{
        value: scores,
        name: currentScenic,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { color: COLORS.primary, width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59,130,246,0.35)' },
            { offset: 1, color: 'rgba(59,130,246,0.05)' }
          ])
        },
        itemStyle: { color: COLORS.primary, borderWidth: 2, borderColor: '#fff' }
      }]
    }],
    animationDuration: 800
  });
}

/* ============================================================
   7. 正负面词云 - ECharts词云图
   ============================================================ */
function renderWordcloudChart(data) {
  if (!data) return;
  var positiveWords = data.positive || [];
  var negativeWords = data.negative || [];

  // 正面词云
  if (charts.wordcloudPositive) {
    charts.wordcloudPositive.setOption({
      tooltip: {
        backgroundColor: '#fff',
        borderColor: COLORS.border,
        borderWidth: 1,
        textStyle: { color: COLORS.text, fontSize: 13 },
        formatter: function(params) {
          return params.name + ': ' + params.value;
        }
      },
      series: [{
        type: 'wordCloud',
        shape: 'circle',
        left: 'center',
        top: 'center',
        width: '90%',
        height: '90%',
        sizeRange: [14, 42],
        rotationRange: [-30, 30],
        rotationStep: 15,
        gridSize: 8,
        drawOutOfBound: false,
        textStyle: {
          fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif',
          fontWeight: 'bold',
          color: function() {
            // 绿色系随机
            var greens = ['#22c55e', '#16a34a', '#4ade80', '#86efac', '#15803d', '#34d399'];
            return greens[Math.floor(Math.random() * greens.length)];
          }
        },
        emphasis: {
          textStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(34,197,94,0.3)',
            color: '#15803d'
          }
        },
        data: positiveWords.map(function(w) {
          return { name: w.name, value: w.value };
        })
      }]
    });
  }

  // 负面词云
  if (charts.wordcloudNegative) {
    charts.wordcloudNegative.setOption({
      tooltip: {
        backgroundColor: '#fff',
        borderColor: COLORS.border,
        borderWidth: 1,
        textStyle: { color: COLORS.text, fontSize: 13 },
        formatter: function(params) {
          return params.name + ': ' + params.value;
        }
      },
      series: [{
        type: 'wordCloud',
        shape: 'circle',
        left: 'center',
        top: 'center',
        width: '90%',
        height: '90%',
        sizeRange: [14, 42],
        rotationRange: [-30, 30],
        rotationStep: 15,
        gridSize: 8,
        drawOutOfBound: false,
        textStyle: {
          fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif',
          fontWeight: 'bold',
          color: function() {
            // 红色系随机
            var reds = ['#ef4444', '#dc2626', '#f87171', '#fca5a5', '#b91c1c', '#f97316'];
            return reds[Math.floor(Math.random() * reds.length)];
          }
        },
        emphasis: {
          textStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(239,68,68,0.3)',
            color: '#b91c1c'
          }
        },
        data: negativeWords.map(function(w) {
          return { name: w.name, value: w.value };
        })
      }]
    });
  }
}

/* ============================================================
   8. 评论长度分布 - 柱状图
   ============================================================ */
function renderCommentLengthChart(data) {
  if (!charts.commentLength || !data) return;
  // API返回: [{range, count}, ...]
  var items = Array.isArray(data) ? data : [];
  var ranges = items.map(function(d) { return d.range; });
  var counts = items.map(function(d) { return d.count; });

  charts.commentLength.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 }
    },
    grid: { left: 50, right: 20, top: 20, bottom: 50 },
    xAxis: {
      type: 'category',
      data: ranges,
      axisLabel: { fontSize: 11, color: COLORS.textLight, rotate: 20 },
      axisLine: { lineStyle: { color: COLORS.border } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 11, color: COLORS.textMuted },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    series: [{
      type: 'bar',
      data: counts,
      barWidth: '50%',
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#8b5cf6' },
          { offset: 1, color: '#c4b5fd' }
        ]),
        borderRadius: [4, 4, 0, 0]
      }
    }],
    animationDuration: 800,
    animationEasing: 'cubicOut'
  });
}

/* ============================================================
   9. 最有用评论 - HTML列表卡片
   ============================================================ */
function renderUsefulComments(data) {
  if (!data) return;
  var list = document.getElementById('usefulCommentsList');
  if (!list) return;

  if (!Array.isArray(data) || data.length === 0) {
    list.innerHTML = '<div style="text-align:center;padding:30px;color:#94a3b8;font-size:.85rem;">暂无有用评论数据</div>';
    return;
  }

  var html = '';
  data.forEach(function(item) {
    var sentiment = item.sentiment || '中性';
    var sentimentCls = sentiment === '正面' ? 'positive' : sentiment === '负面' ? 'negative' : 'neutral';
    var sentimentEmoji = sentiment === '正面' ? '😊' : sentiment === '负面' ? '😞' : '😐';

    html += '<div class="useful-comment-item">';
    html += '  <div class="useful-comment-content">' + escapeHtml(item.content || '') + '</div>';
    html += '  <div class="useful-comment-meta">';
    html += '    <span class="useful-comment-badge ' + sentimentCls + '">' + sentimentEmoji + ' ' + sentiment + '</span>';
    html += '    <span class="useful-comment-stat">⭐ 评分: ' + (item.score || '--') + '</span>';
    html += '    <span class="useful-comment-stat">👍 有用: ' + (item.useful_count || item.useful || 0) + '</span>';
    html += '  </div>';
    html += '</div>';
  });

  list.innerHTML = html;
}

/* HTML转义 */
function escapeHtml(text) {
  var div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/* ============================================================
   10. 景区综合对比 - 雷达图+表格
   ============================================================ */
function renderComparisonChart(data) {
  if (!charts.comparison || !data) return;
  // API返回: [{name, total, avg_score, positive_pct, dimension_scores:{风景:0.52,...}}, ...]
  var items = Array.isArray(data) ? data : [];
  if (items.length === 0) return;

  var scenics = items.map(function(d) { return d.name; });
  // 获取维度名称（从第一个景区的dimension_scores取key）
  var firstDims = items[0].dimension_scores || {};
  var dimensions = Object.keys(firstDims);
  if (dimensions.length === 0) return;

  // 为每个景区生成不同颜色
  var palette = [
    '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1',
    '#84cc16', '#e11d48'
  ];

  // 构建雷达图数据 - dimension_scores转为0-10分制
  var seriesData = items.map(function(item, idx) {
    var ds = item.dimension_scores || {};
    var values = dimensions.map(function(dim) {
      return Math.round((ds[dim] || 0) * 10 * 10) / 10; // 0-10分制
    });
    return {
      value: values,
      name: item.name,
      symbol: 'circle',
      symbolSize: 5,
      lineStyle: { width: 2, color: palette[idx % palette.length] },
      areaStyle: { color: palette[idx % palette.length], opacity: 0.08 },
      itemStyle: { color: palette[idx % palette.length], borderWidth: 2, borderColor: '#fff' }
    };
  });

  charts.comparison.setOption({
    tooltip: {
      backgroundColor: '#fff',
      borderColor: COLORS.border,
      borderWidth: 1,
      textStyle: { color: COLORS.text, fontSize: 13 }
    },
    legend: {
      data: scenics,
      top: 0,
      textStyle: { fontSize: 11, color: COLORS.textLight },
      type: 'scroll'
    },
    radar: {
      indicator: dimensions.map(function(d) { return { name: d, max: 10 }; }),
      shape: 'polygon',
      splitNumber: 5,
      center: ['50%', '55%'],
      radius: '60%',
      axisName: { color: COLORS.textLight, fontSize: 12, fontWeight: 500 },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
      splitArea: { areaStyle: { color: ['#fafbff', '#f1f5f9'] } },
      axisLine: { lineStyle: { color: '#e2e8f0' } }
    },
    series: [{
      type: 'radar',
      data: seriesData
    }],
    animationDuration: 800
  });

  // 渲染对比表格 - 传入items数组
  renderComparisonTable(scenics, dimensions, items);
}

/* 渲染对比表格 */
function renderComparisonTable(scenics, dimensions, scores) {
  var wrap = document.getElementById('comparisonTableWrap');
  if (!wrap) return;

  // scores现在是items数组，每个item有dimension_scores
  var items = scores;

  var html = '<table class="comparison-table">';
  html += '<thead><tr><th>景区</th><th>评论数</th><th>好评率</th>';
  dimensions.forEach(function(dim) {
    html += '<th>' + dim + '</th>';
  });
  html += '</tr></thead><tbody>';

  items.forEach(function(item) {
    var ds = item.dimension_scores || {};
    html += '<tr><td>' + item.name + '</td>';
    html += '<td>' + (item.total || 0) + '</td>';
    html += '<td>' + (item.positive_pct || 0) + '%</td>';
    dimensions.forEach(function(dim) {
      var val = Math.round((ds[dim] || 0) * 10 * 10) / 10;
      html += '<td>' + val + '</td>';
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
}