import { useState, useEffect } from 'react';

const useDatasetStats = () => {
  const [stats, setStats] = useState({
    totalSentences: 0,
    totalLabels: 0,
    distribution: {
      emotional_progress: 0,
      emotional_distress: 0,
      danger: 0,
      emotionally_intense: 0
    }
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAndAnalyzeData = async () => {
      try {
        const response = await fetch('/data/evaluation_data.json');
        const data = await response.json();

        // Calculate statistics
        const totalSentences = data.length;
        let totalLabels = 0;
        const labelCounts = {
          emotional_progress: 0,
          emotional_distress: 0,
          danger: 0,
          emotionally_intense: 0
        };

        // Count labels
        data.forEach(item => {
          if (item.tags) {
            totalLabels += item.tags.length;
            item.tags.forEach(tag => {
              if (tag.label in labelCounts) {
                labelCounts[tag.label]++;
              }
            });
          }
        });

        // Calculate percentages
        const distribution = Object.entries(labelCounts).reduce((acc, [key, count]) => {
          acc[key] = {
            count,
            percentage: ((count / totalLabels) * 100).toFixed(2)
          };
          return acc;
        }, {});

        setStats({
          totalSentences,
          totalLabels,
          distribution
        });
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchAndAnalyzeData();
  }, []);

  return { stats, loading, error };
};

export default useDatasetStats; 