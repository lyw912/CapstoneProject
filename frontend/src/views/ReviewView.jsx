import { Alert, Button, Progress, Space, Timeline } from 'antd';
import { motion } from 'framer-motion';
import {
  FileTextOutlined,
  CloudDownloadOutlined,
  BookOutlined,
  FilePdfOutlined
} from '@ant-design/icons';
import SectionTitle from '../components/SectionTitle';
import ReviewEditor from '../components/ReviewEditor';

export default function ReviewView({ output, reportTask, reportHtml, setReportHtml, annotations, setAnnotations, theme, reportEvents, generateReport }) {
  return (
    <motion.section key="review" className="review-page" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
      <div className="review-header studio-card">
        <SectionTitle eyebrow="Report board" title="Edit, highlight, and annotate the final narrative" />
        <Space wrap>
          <Button icon={<FileTextOutlined />} onClick={generateReport}>Generate Report</Button>
          <Button icon={<CloudDownloadOutlined />} disabled={!reportTask?.task_id} href={reportTask?.task_id ? `/api/report/download/${reportTask.task_id}` : undefined}>HTML</Button>
          <Button icon={<BookOutlined />} disabled={!reportTask?.task_id} href={reportTask?.task_id ? `/api/report/export/md/${reportTask.task_id}` : undefined}>Markdown</Button>
          <Button icon={<FilePdfOutlined />} disabled={!reportTask?.task_id} href={reportTask?.task_id ? `/api/report/export/pdf/${reportTask.task_id}` : undefined}>PDF</Button>
        </Space>
      </div>
      {reportTask?.status === 'running' && <Alert type="info" showIcon message="Report generation is running" description={<Progress percent={reportTask.progress || 0} strokeColor={theme.primary} />} />}
      <ReviewEditor output={output} reportHtml={reportHtml} onReportHtmlChange={setReportHtml} annotations={annotations} setAnnotations={setAnnotations} />
      <div className="studio-card event-card">
        <SectionTitle eyebrow="Generation stream" title="Recent writing events" />
        <Timeline items={(reportEvents.length ? reportEvents : [{ message: 'No report events yet', type: 'idle' }]).slice(0, 8).map((item) => ({ color: item.type === 'error' ? 'red' : item.type === 'warning' ? 'gold' : 'green', children: <span>{item.message}</span> }))} />
      </div>
    </motion.section>
  );
}
