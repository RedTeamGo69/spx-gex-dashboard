-- PostgreSQL enforcement: application retries cannot replace frozen evidence.
CREATE OR REPLACE FUNCTION ft_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Forward study evidence is immutable; append a correction';
END;
$$;
DROP TRIGGER IF EXISTS ft_forecasts_immutable ON ft_forecasts;
CREATE TRIGGER ft_forecasts_immutable BEFORE UPDATE OR DELETE ON ft_forecasts FOR EACH ROW EXECUTE FUNCTION ft_reject_mutation();
DROP TRIGGER IF EXISTS ft_inputs_immutable ON ft_inputs;
CREATE TRIGGER ft_inputs_immutable BEFORE UPDATE OR DELETE ON ft_inputs FOR EACH ROW EXECUTE FUNCTION ft_reject_mutation();
DROP TRIGGER IF EXISTS ft_observations_immutable ON ft_observations;
CREATE TRIGGER ft_observations_immutable BEFORE UPDATE OR DELETE ON ft_observations FOR EACH ROW EXECUTE FUNCTION ft_reject_mutation();
DROP TRIGGER IF EXISTS ft_scores_immutable ON ft_scores;
CREATE TRIGGER ft_scores_immutable BEFORE UPDATE OR DELETE ON ft_scores FOR EACH ROW EXECUTE FUNCTION ft_reject_mutation();
DROP TRIGGER IF EXISTS ft_studies_immutable ON ft_studies;
CREATE TRIGGER ft_studies_immutable BEFORE UPDATE OR DELETE ON ft_studies FOR EACH ROW EXECUTE FUNCTION ft_reject_mutation();
CREATE OR REPLACE FUNCTION ft_guard_slot_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.slot_id, NEW.study_id, NEW.week_start, NEW.ticker, NEW.model, NEW.model_version, NEW.cohort)
       IS DISTINCT FROM ROW(OLD.slot_id, OLD.study_id, OLD.week_start, OLD.ticker, OLD.model, OLD.model_version, OLD.cohort)
       OR (OLD.status IN ('captured','missed') AND NEW.status <> OLD.status) THEN
        RAISE EXCEPTION 'Forward study slot identity and terminal admission are immutable';
    END IF;
    RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS ft_slots_identity ON ft_slots;
CREATE TRIGGER ft_slots_identity BEFORE UPDATE ON ft_slots FOR EACH ROW EXECUTE FUNCTION ft_guard_slot_identity();
