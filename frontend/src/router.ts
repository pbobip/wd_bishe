import { createRouter, createWebHistory } from 'vue-router'

const HistoryView = () => import('./views/HistoryView.vue')
const ProjectManagementView = () => import('./views/ProjectManagementView.vue')
const ResultsView = () => import('./views/ResultsView.vue')
const RunDetailView = () => import('./views/RunDetailView.vue')
const RunStatisticsView = () => import('./views/RunStatisticsView.vue')
const TaskConfigView = () => import('./views/TaskConfigView.vue')
const TaskRunView = () => import('./views/TaskRunView.vue')

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/projects',
      name: 'projects',
      component: ProjectManagementView,
      meta: {
        sectionKey: 'projects',
        sectionTitle: '项目创建',
        sectionSubtitle: '创建、查看和管理项目，为不同实验批次或研究方向分组留痕。',
      },
    },
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
        sectionSubtitle: '导入图像并配置分割与预处理，再进入后处理与运行。',
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
        sectionTitle: '后处理',
        sectionSubtitle: '配置后处理、统计导出并查看上传与执行状态。',
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
      component: ResultsView,
      meta: {
        sectionKey: 'results',
        sectionTitle: '结果展示',
        sectionSubtitle: '集中查看已完成任务、关键统计和最近输出，适合答辩展示与结果核查。',
      },
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
        sectionKey: 'results',
        sectionTitle: '结果展示',
        sectionSubtitle: '查看当前任务的步骤进度、分割图像、统计摘要和导出文件。',
      },
    },
    {
      path: '/runs/:id/statistics',
      name: 'run-statistics',
      component: RunStatisticsView,
      props: true,
      meta: {
        sectionKey: 'results',
        sectionTitle: '结果展示',
        sectionSubtitle: '按统计量查看 Vf、颗粒数、面积、尺寸分布和颗粒级明细。',
      },
    },
  ],
})
