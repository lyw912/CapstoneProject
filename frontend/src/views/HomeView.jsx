import { motion } from 'framer-motion';
import {
  SafetyCertificateOutlined,
  SearchOutlined,
  ClockCircleOutlined,
  WarningOutlined
} from '@ant-design/icons';
import SignalGraph from '../components/SignalGraph';
import MiniMetric from '../components/MiniMetric';
import { percentText, compactNumber, durationText } from '../utils/helpers';

export default function HomeView({ output, coordinatorTask, theme, risks, queryAgent, synthesis, setActive }) {
  return (
    <motion.section key="command" className="home-stage" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
      <SignalGraph output={output} task={coordinatorTask} theme={theme} onOpen={() => setActive('intelligence')} />
      <div className="metric-orbit">
        <MiniMetric icon={<SafetyCertificateOutlined />} label="Trust" value={percentText(synthesis.overall_confidence)} onClick={() => setActive('intelligence')} />
        <MiniMetric icon={<SearchOutlined />} label="Proof" value={compactNumber(queryAgent.total_sources)} onClick={() => setActive('evidence')} />
        <MiniMetric icon={<ClockCircleOutlined />} label="Time" value={durationText(output.pipeline_duration_seconds).replace('No run yet', 'Idle')} onClick={() => setActive('control')} />
        <MiniMetric icon={<WarningOutlined />} label="Risk" value={compactNumber(risks.length)} onClick={() => setActive('intelligence')} />
      </div>
    </motion.section>
  );
}
