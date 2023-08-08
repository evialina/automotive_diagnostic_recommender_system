import { Datagrid, List, TextField, DateField, NumberField, ShowButton } from 'react-admin';

export const ConsultationList = () => (
    <List>
        <Datagrid>
            <DateField source='consultation_date' />
            <TextField source='anonymised_vin' />
            <NumberField source='warranty_records' />
            <ShowButton />
        </Datagrid>
    </List>
)
