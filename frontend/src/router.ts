import { createRouter, createWebHistory } from 'vue-router'

const HistoryView = () => import('./views/HistoryView.vue')
const RunDetailView = () => import('./views/RunDetailView.vue')
const RunStatisticsView = () => import('./views/RunStatisticsView.vue')
const TaskConfigView = () => import('./views/TaskConfigView.vue')
const TaskRunView = () => import('./views/TaskRunView.vue')

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/tasks/create',
    },
    {
      path: '/tasks/create',
      name: 'task-create',
      component: TaskConfigView,
      meta: {
        sectionKey: 'task-create',
        sectionTitle: '任务创建',
        sectionSubtitle: '导入图像、逐图标定并启动主分割。',
      },
    },
    {
      path: '/processing',
      redirect: (to) => ({
        path: '/tasks/create',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/postprocess',
      name: 'postprocess',
      component: TaskRunView,
      meta: {
        sectionKey: 'postprocess',
        sectionTitle: '结果展示与后处理',
        sectionSubtitle: '先核查当前分割结果，再决定是否应用后处理。',
      },
    },
    {
      path: '/workflow/task',
      redirect: (to) => ({
        path: '/tasks/create',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/workflow/config',
      redirect: (to) => ({
        path: '/tasks/create',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/workflow/draft',
      redirect: (to) => ({
        path: '/tasks/create',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/workflow/run',
      redirect: (to) => ({
        path: '/postprocess',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/workflow/postprocess',
      redirect: (to) => ({
        path: '/postprocess',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/run-center',
      redirect: (to) => ({
        path: '/postprocess',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/results',
      name: 'results',
      redirect: (to) => ({
        path: '/history',
        query: to.query,
        hash: to.hash,
      }),
    },
    {
      path: '/history',
      name: 'history',
      component: HistoryView,
      meta: {
        sectionKey: 'history',
        sectionTitle: '历史记录',
        sectionSubtitle: '回溯全部运行任务的状态、配置和时间轴。',
      },
    },
    {
      path: '/runs/:id',
      name: 'run-detail',
      component: RunDetailView,
      props: true,
      meta: {
        sectionKey: 'postprocess',
        sectionTitle: '结果展示与后处理',
        sectionSubtitle: '查看当前任务的分割结果、步骤轨迹与后处理入口。',
      },
    },
    {
      path: '/runs/:id/statistics',
      name: 'run-statistics',
      component: RunStatisticsView,
      props: true,
      meta: {
        sectionKey: 'statistics',
        sectionTitle: '统计分析',
        sectionSubtitle: '基于当前确认结果查看统计卡片、图表与明细。',
      },
    },
  ],
})
